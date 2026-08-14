# Streaming Flow Policy in LeRobot

This implementation follows the stabilized deterministic action-space SFP equations from
the official Push-T notebook:

```text
a(t) = xi(t) + sigma_0 exp(-k t) epsilon
v(t) = dxi(t)/dt - k (a(t) - xi(t))
```

LeRobot samples observations and actions in one aligned window. The first
`n_obs_steps - 1` actions belong to the observation history, so training removes them before
interpolating the current/future action curve. Inference emits the state before each Euler
step and carries the post-chunk integration state into the next chunk.

The defaults `sigma_0=0.4`, `k=10`, and one Euler step per emitted action match the latest
official Push-T notebook. `integration_steps` adds Euler substeps per emitted action interval
when a deployment needs a more accurate integration/latency trade-off.

Reference: https://github.com/siddancha/streaming-flow-policy
