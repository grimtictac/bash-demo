This is the actual hand-writen report :)

My approach was to feed the code to the AI and have it one-shot as much as it could but to then go back and actually take ownership of it.
Nobody wants to just be blasting a whole bunch of stuff into prod!

I definitely didn't have enough time to get to everything, so I've focused on 3 key areas where I can:
1) demonstrate that I understand the problem (by reproducing it)
2) demonstrate the fix (see the demos folder)
3) explain how the fix works and why it's valid (some of that is captured in DECISIONS.md, but it's also a but verbose. I'll try to be more brief here.)

Also please make sure to watch the demo video here (3m 43s):
https://drive.google.com/file/d/1MAUZX8JbzD5jCC090T3j_p9LBQ364Clg/view
Sometimes it's just quicker to show rather than type...

# Concurrency and serialization
service.predict is a blocking call inside an async handler. This was hodling the event loop which meant that no other calls could       
  complete. This is the "serialization" problem, reqeusts are handled one by one. Moving the blocking call to a worker thread released the event loop to process other requests concurrently.

The demos/serialisation folder contains a demonstration of the slow (serial) server and the fast (parallel) one.
(technically parallel is not accurate, it's concurrent because it's an event loop)

# Output validation
The key insight for this exercise is that changing the model is not a code deployment, the service will remain running the whole time. So we don't have a startup time or CI pipeline or tests to check things before moving from one version to the next.
As implied in the demos/rollover folder we need to hot-swap which means being particularly conscious of making sure we have done something valid to avoid silent failure.

What I've added is really just the bare minimum. Make sure that the mode is retuning a plausible number.
That's "shape" but it's not semantics.

What would be good for future consideration:
- have one or more golden examples, that can be loaded before switching the model over to make sure there are no regressions
- it might be overkill but we could consider separating load from rollover. One call loads the model but leaves it staged, then we can add a param or header to the query that selects which model to use, we manually smoke test the staged model and when we are happy apply the switch. We can even keep the old model in memory for faster rollback (if the load takes any significant amount of time) and the deleting it can be its own seperate step.

In the real world, I would expect this kind of thing to be handled more with downstream pods or something, and this service would be more of a loadbalancer. We can discuss in more detail though.

# Rollover
The idea here is that switching from one model to the next should not stall the service.
That is demonstrated in the demo files.

