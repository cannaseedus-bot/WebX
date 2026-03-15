/**
 * @fileoverview Work scheduler for KUHUL runtime tasks.
 *
 * Provides a simple FIFO queue with priority support for scheduling
 * asynchronous tasks (shader dispatches, host callbacks, etc.).
 *
 * @module kuhul/runtime/scheduler
 */

// ------------------------------------------------------------------ //
// Task
// ------------------------------------------------------------------ //

/**
 * @typedef {{ fn: () => Promise<*>, priority: number, name: string }} Task
 */

// ------------------------------------------------------------------ //
// Scheduler
// ------------------------------------------------------------------ //

/** Schedules and runs async tasks in priority order. */
export class Scheduler {
  constructor() {
    /** @type {Task[]} */
    this._queue   = [];
    this._running = false;
  }

  /**
   * Add a task to the scheduler.
   *
   * @param {{ fn: () => Promise<*>, priority?: number, name?: string }} task
   * @returns {Scheduler}
   */
  schedule(task) {
    this._queue.push({
      fn:       task.fn,
      priority: task.priority ?? 0,
      name:     task.name     ?? 'anonymous',
    });
    // Keep queue sorted: higher priority runs first
    this._queue.sort((a, b) => b.priority - a.priority);
    return this;
  }

  /**
   * Run all scheduled tasks in priority order, clearing the queue.
   *
   * @returns {Promise<*[]>} Results from all tasks
   */
  async run() {
    if (this._running) {
      throw new Error('Scheduler is already running');
    }
    this._running = true;
    const results = [];

    while (this._queue.length > 0) {
      const task = this._queue.shift();
      try {
        const result = await task.fn();
        results.push({ name: task.name, result, error: null });
      } catch (err) {
        results.push({ name: task.name, result: null, error: err });
      }
    }

    this._running = false;
    return results;
  }

  /**
   * Number of tasks currently in the queue.
   * @returns {number}
   */
  get size() { return this._queue.length; }

  /** Clear all pending tasks without running them. */
  clear() { this._queue = []; }
}
