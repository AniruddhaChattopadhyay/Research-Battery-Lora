| Size | Decision model | Backbone | Task | n | Acc dec | Acc base | Gain [95% CI] | ECE dec | ECE base | ECE-TS dec | ECE-TS base |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B | Tev1-4B | local/qwen3.5-4b | boolq | 500 | 89.2 | 81.2 | +8.0 | 2.8 | 6.8 | 3.4 | 3.8 |
| 4B | Tev1-4B | local/qwen3.5-4b | aegis | 498 | 76.9 | 72.1 | +4.8 | 10.8 | 16.8 | 5.8 | 9.3 |
| 4B | Tev1-4B | local/qwen3.5-4b | civil | 3500 | 80.3 | 80.6 | -0.3 | 3.0 | 5.3 | 3.1 | 2.5 |
| 4B | Tev1-4B | local/qwen3.5-4b | arxiv | 500 | 81.2 | 83.4 | -2.2 | 3.8 | 6.0 | 4.6 | 4.8 |
| 4B | Tev1-4B | local/qwen3.5-4b | when2call | 500 | 73.4 | 54.2 | +19.2 | 10.3 | 24.2 | 9.2 | 12.8 |
| 4B | Tev1-4B | local/qwen3.5-4b | judgebench | 486 | 29.8 | 29.6 | +0.2 | 49.2 | 50.1 | 22.4 | 22.4 |
| 4B | Tev1-4B | local/qwen3.5-4b | helpsteer2 | 2500 | 39.2 | 34.1 | +5.1 | 12.4 | 20.2 | 8.0 | 10.4 |
| 4B | Tev1-4B | local/qwen3.5-4b | all | 8484 | 67.2 | 62.2 | +5.0 [+3.7, +6.2] | 13.2 | 18.5 | 8.1 | 9.4 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | boolq | 500 | 90.0 | 84.0 | +6.0 | 6.5 | 7.8 | 3.1 | 5.9 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | aegis | 500 | 83.0 | 77.8 | +5.2 | 7.8 | 17.4 | 5.0 | 7.0 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | civil | 3500 | 87.9 | 80.2 | +7.6 | 2.3 | 8.8 | 2.1 | 2.4 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | arxiv | 500 | 87.8 | 87.2 | +0.6 | 2.9 | 3.8 | 4.9 | 4.8 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | when2call | 500 | 75.6 | 60.6 | +15.0 | 16.4 | 20.8 | 6.3 | 8.7 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | judgebench | 500 | 35.8 | 30.4 | +5.4 | 41.5 | 46.7 | 16.0 | 21.6 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | helpsteer2 | 2500 | 48.4 | 35.6 | +12.8 | 8.3 | 30.9 | 5.1 | 10.4 |
| 9B | Microsoft-Decision-1 | qwen/qwen3.5-9b | all | 8500 | 72.6 | 65.1 | +7.5 [+6.2, +8.7] | 12.2 | 19.4 | 6.1 | 8.7 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | boolq | 500 | 92.6 | 80.8 | +11.8 | 1.8 | 12.0 | 3.1 | 7.0 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | aegis | 500 | 86.2 | 83.6 | +2.6 | 7.0 | 10.4 | 3.3 | 6.3 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | civil | 3500 | 90.7 | 81.3 | +9.4 | 1.7 | 10.9 | 1.8 | 4.7 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | arxiv | 500 | 86.6 | 88.8 | -2.2 | 4.0 | 5.6 | 7.1 | 3.4 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | when2call | 500 | 74.8 | 75.0 | -0.2 | 17.1 | 14.6 | 10.2 | 7.2 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | judgebench | 500 | 53.4 | 35.2 | +18.2 | 26.5 | 40.3 | 8.1 | 16.7 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | helpsteer2 | 2500 | 51.5 | 34.8 | +16.7 | 4.3 | 14.4 | 5.5 | 3.6 |
| 27B | pplx-decider v1.1 | qwen/qwen3.8-27b | all | 8500 | 76.5 | 68.5 | +8.0 [+6.8, +9.2] | 8.9 | 15.4 | 5.6 | 7.0 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | boolq | 500 | 87.4 | 86.2 | +1.2 | 8.1 | 12.9 | 6.5 | 3.5 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | aegis | 500 | 83.8 | 84.4 | -0.6 | 12.3 | 13.9 | 3.8 | 3.4 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | civil | 3500 | 82.0 | 90.4 | -8.4 | 11.8 | 4.4 | 3.7 | 2.8 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | arxiv | 500 | 88.6 | 90.8 | -2.2 | 5.6 | 5.1 | 6.0 | 3.7 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | when2call | 500 | 69.6 | 76.0 | -6.4 | 15.7 | 23.2 | 8.2 | 7.3 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | judgebench | 493 | 49.1 | 83.4 | -34.3 | 28.0 | 14.6 | 3.1 | 4.5 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | helpsteer2 | 2500 | 31.6 | 39.2 | -7.6 | 41.2 | 22.6 | 1.8 | 3.8 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna | all | 8493 | 70.3 | 78.6 | -8.3 [-9.4, -7.2] | 17.5 | 13.8 | 4.7 | 4.2 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | boolq | 500 | 87.4 | 86.2 | +1.2 | 8.1 | 12.2 | 6.5 | 4.8 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | aegis | 500 | 83.8 | 84.0 | -0.2 | 12.3 | 14.7 | 3.8 | 3.8 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | civil | 3500 | 82.0 | 90.3 | -8.3 | 11.8 | 3.9 | 3.7 | 2.8 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | arxiv | 500 | 88.6 | 88.0 | +0.6 | 5.6 | 5.9 | 6.0 | 5.5 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | when2call | 500 | 69.6 | 70.0 | -0.4 | 15.7 | 29.6 | 8.2 | 2.8 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | judgebench | 495 | 48.9 | 41.2 | +7.7 | 28.2 | 55.5 | 3.3 | 17.2 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | helpsteer2 | 2500 | 31.6 | 39.7 | -8.2 | 41.2 | 21.9 | 1.8 | 3.6 |
| closed | OpenAI Decisions (Luna) | openai/gpt-6-luna@none | all | 8495 | 70.3 | 71.3 | -1.1 [-2.1, -0.0] | 17.6 | 20.6 | 4.8 | 5.8 |
