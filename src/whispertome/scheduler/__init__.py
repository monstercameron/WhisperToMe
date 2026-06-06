"""Future-facing scheduled events: reminders and recorded tool-sequence workflows that
fire at a due time. The core (model/clock/executor) is audio/NPU-free and unit-testable;
the SchedulerThread wires it to the running voice loop."""
