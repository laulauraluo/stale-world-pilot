| model | procedure | success, static | success, changed | stale action, changed | detected, changed | false alarm, static | asked to look |
|---|---|---|---|---|---|---|---|
| GLM-4.1V-9B-Thinking | memory | 99.5% [97.2, 99.9] | 8.5% [5.4, 13.2] | 88.0% [82.8, 91.8] | 2.5% [1.1, 5.7] | 2.0% [0.8, 5.0] | 3.5% [1.7, 7.0] |
| GLM-4.1V-9B-Thinking | recheck | 97.5% [94.3, 98.9] | 76.5% [70.2, 81.8] | 4.0% [2.0, 7.7] | 82.0% [76.1, 86.7] | 3.0% [1.4, 6.4] | n/a |
| GLM-4.1V-9B-Thinking | facts | 99.5% [97.2, 99.9] | 96.0% [92.3, 98.0] | 1.5% [0.5, 4.3] | 96.0% [92.3, 98.0] | 4.0% [2.0, 7.7] | n/a |
| GLM-4.1V-9B-Thinking | replay | 100.0% [98.1, 100.0] | 93.5% [89.2, 96.2] | 1.5% [0.5, 4.3] | 91.5% [86.8, 94.6] | 3.5% [1.7, 7.0] | n/a |

Each cell: rate over n = 200 episodes, with a 95% Wilson interval in brackets.
09:00 descriptions that call an aisle blocked (none is at 09:00): GLM-4.1V-9B-Thinking 0/200.
