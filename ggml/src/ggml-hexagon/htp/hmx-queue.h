#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>

#include <hexagon_types.h>
#include <qurt_thread.h>
#include <qurt_futex.h>
#include <HAP_farf.h>

#include "hex-utils.h"
#include "hex-profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// V79 used to fall straight into qurt_futex_wait after every drain (poll count 1), so
// every subsequent push cost a futex wake plus a thread wakeup. Profiling showed that as
// ~1150 idle cycles paid once per HMX job, and ~45% of all fully-idle gaps resumed into
// HMX_COMP. A short spin removes it. 64 is the measured knee: 64/512/2000 are within
// noise of each other, and a shorter spin steals fewer cycles from the six HVX workers
// that share the hardware threads.
#if __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_COUNT 2000
#else
#define HMX_QUEUE_POLL_COUNT 64
#endif

// The push-side futex wake is a QuRT syscall on the critical path of every HMX
// dispatch, and the consumer is almost never actually asleep during an op (see
// hmx_queue_push). Set to 0 to restore the unconditional wake -- the control for
// attributing any measured change to this mechanism rather than to the pause grades.
#ifndef HMX_QUEUE_COND_WAKE
#define HMX_QUEUE_COND_WAKE 1
#endif

typedef void (*hmx_queue_func)(void *);

// Dummy funcs used as signals
enum hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

struct hmx_queue_desc {
    hmx_queue_func   func;
    void *           data;
    atomic_uint      done;
};

struct hmx_queue_s {
    struct hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned int     idx_pop;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    atomic_uint      sleeping;  // 1 only while the consumer is inside qurt_futex_wait
    qurt_thread_t    thread;
    void *           stack;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    struct htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

typedef struct hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t stack_size);
size_t      hmx_queue_alignof(void);
hmx_queue_t hmx_queue_init(void * ptr, size_t capacity, uint32_t stack_size, uint32_t hap_rctx, struct htp_thread_trace * trace);
void        hmx_queue_free(hmx_queue_t q);

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    unsigned int ir = atomic_load(&q->idx_read);
    unsigned int iw = atomic_load(&q->idx_write);

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(HIGH, "hmx-queue-push: queue is full\n");
        return false;
    }

    atomic_store(&d.done, 0);

    FARF(HIGH, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    atomic_fetch_add(&q->seqn, 1);

    // The wake is a QuRT syscall (~250 cyc, derived from idle-before-O_PROC) on the
    // critical path of all 576 pushes per per-query-block flash-attn op. With
    // HMX_QUEUE_POLL_COUNT 64 the consumer spins ~16k cycles after each drain, which
    // exceeds every inter-push interval in an FA op -- so in the steady state it never
    // has a waiter to wake.
    //
    // No lost wakeup: the seqn increment above is seq_cst and precedes this load; the
    // consumer stores `sleeping` seq_cst BEFORE re-comparing seqn. If we read
    // sleeping == 0, the consumer's compare is ordered after our increment and it
    // returns/skips without sleeping.
    if (!HMX_QUEUE_COND_WAKE || atomic_load(&q->sleeping)) {
        qurt_futex_wake(&q->seqn, 1);
    }

    return true;
}

static inline bool hmx_queue_signal(hmx_queue_t q, enum hmx_queue_signal sig) {
    return hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) sig, NULL));
}

static inline bool hmx_queue_empty(hmx_queue_t q) {
    return q->idx_pop == atomic_load(&q->idx_write);
}

static inline uint32_t hmx_queue_depth(hmx_queue_t q) {
    return (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

static inline uint32_t hmx_queue_capacity(hmx_queue_t q) {
    return q->capacity;
}

static inline struct hmx_queue_desc hmx_queue_pop_one(hmx_queue_t q) {
    unsigned int ip = q->idx_pop;
    unsigned int iw = atomic_load(&q->idx_write);

    struct hmx_queue_desc rd = { NULL, NULL };
    if (ip == iw) {
        return rd;
    }

    // Deliberately the LONG pause, unlike the bounded rendezvous spins in work-queue.c.
    // This wait is a whole HMX tile-matmul, not a handoff: several call sites push a job
    // and immediately pop it with nothing overlapped (matmul-ops.c:2843, :2781, :3056,
    // :3314; flash-attn-ops.c:2804, :2860, :2933), and those are the dense and
    // shared-selection paths. Worse, the waiter competes for a hardware thread with the
    // HMX thread it is waiting on -- n_hvx workers plus the HMX thread is one more
    // runnable thread than there are hardware threads -- so a tighter spin here steals
    // cycles from the very job it is waiting for.
    struct hmx_queue_desc * d = &q->desc[ip];
    while (!atomic_load(&d->done)) {
        FARF(HIGH, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_pop = (ip + 1) & q->idx_mask;

    FARF(HIGH, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    return rd;
}

static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    while (1) {
        struct hmx_queue_desc d = hmx_queue_pop_one(q);

        uint32_t sig = (uint32_t) d.func;
        if (sig && sig <= HMX_QUEUE_KILL)
            continue;

        return d;
    }
}

static inline void hmx_queue_flush(hmx_queue_t q) {
    while (hmx_queue_pop_one(q).func != NULL) ;
}

static inline void hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

static inline void hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
