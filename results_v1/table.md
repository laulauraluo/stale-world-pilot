| model | procedure | success, static | success, changed | stale action, changed | detected, changed | false alarm, static | asked to look |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | memory | 100.0% [98.1, 100.0] | 9.5% [6.2, 14.4] | 90.5% [85.6, 93.8] | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.9] |
| Qwen3-VL-8B | recheck | 100.0% [98.1, 100.0] | 43.5% [36.8, 50.4] | 56.5% [49.6, 63.2] | 71.5% [64.9, 77.3] | 0.0% [0.0, 1.9] | n/a |
| Qwen3-VL-8B | replay | 100.0% [98.1, 100.0] | 100.0% [98.1, 100.0] | 0.0% [0.0, 1.9] | 90.5% [85.6, 93.8] | 1.5% [0.5, 4.3] | n/a |
| Mistral-Small-3.2-24B | memory | 100.0% [98.1, 100.0] | 24.0% [18.6, 30.4] | 76.0% [69.6, 81.4] | 1.5% [0.5, 4.3] | 0.0% [0.0, 1.9] | 3.5% [1.7, 7.0] |
| Mistral-Small-3.2-24B | recheck | 100.0% [98.1, 100.0] | 77.0% [70.7, 82.3] | 23.0% [17.7, 29.3] | 56.0% [49.1, 62.7] | 0.0% [0.0, 1.9] | n/a |
| Mistral-Small-3.2-24B | replay | 100.0% [98.1, 100.0] | 98.5% [95.7, 99.5] | 1.5% [0.5, 4.3] | 73.0% [66.5, 78.7] | 0.0% [0.0, 1.9] | n/a |

Each cell: rate over n = 200 episodes, with a 95% Wilson interval in brackets.
09:00 descriptions that call an aisle blocked (none is at 09:00): Qwen3-VL-8B 0/200, Mistral-Small-3.2-24B 137/200.
