done typesafe/jev-1.14
done openai/gpt-6-luna-decisions
done microsoft/microsoft-decision-1
done perplexity/pplx-decider-v1.1-27b
done cloudflare/clef
done liquid/d1
done inception/mercury-decide

| First | Second | Task | Acc 1 | Acc 2 | Acc cascade | Escalated | Cost ratio (cascade / 2) |
|---|---|---|---:|---:|---:|---:|---:|
| Jev 1.14 | GPT-6 Sol | boolq | 89.8 | 89.0 | 88.4 | 24% | 0.29 |
| Jev 1.14 | GPT-6 Sol | aegis | 85.2 | 83.4 | 85.3 | 0% | 0.02 |
| Jev 1.14 | GPT-6 Sol | civil | 79.7 | 90.9 | 90.9 | 94% | 0.96 |
| Jev 1.14 | GPT-6 Sol | banking77 | 79.2 | 83.8 | 82.2 | 27% | 0.32 |
| Jev 1.14 | GPT-6 Sol | arxiv | 89.4 | 88.6 | 89.3 | 0% | 0.01 |
| Jev 1.14 | GPT-6 Sol | when2call | 74.2 | 76.6 | 76.5 | 12% | 0.15 |
| Jev 1.14 | GPT-6 Sol | judgebench | 51.2 | 91.0 | 90.8 | 98% | 1.00 |
| Jev 1.14 | GPT-6 Sol | helpsteer2 | 39.6 | 37.7 | 39.6 | 0% | 0.01 |
| Jev 1.14 | Claude Sonnet 5.5 | boolq | 89.8 | 90.4 | 89.8 | 0% | 0.02 |
| Jev 1.14 | Claude Sonnet 5.5 | aegis | 84.9 | 80.4 | 85.0 | 0% | 0.01 |
| Jev 1.14 | Claude Sonnet 5.5 | civil | 79.7 | 91.8 | 91.6 | 92% | 0.92 |
| Jev 1.14 | Claude Sonnet 5.5 | banking77 | 79.2 | 84.2 | 84.0 | 32% | 0.36 |
| Jev 1.14 | Claude Sonnet 5.5 | arxiv | 89.4 | 92.4 | 91.6 | 31% | 0.31 |
| Jev 1.14 | Claude Sonnet 5.5 | when2call | 74.2 | 78.4 | 78.5 | 30% | 0.30 |
| Jev 1.14 | Claude Sonnet 5.5 | judgebench | 51.5 | 80.1 | 80.1 | 96% | 0.99 |
| Jev 1.14 | Claude Sonnet 5.5 | helpsteer2 | 39.7 | 40.3 | 39.6 | 50% | 0.50 |
| Jev 1.14 | Gemini 3.8 Flash | boolq | 89.8 | 79.6 | 89.8 | 0% | 0.03 |
| Jev 1.14 | Gemini 3.8 Flash | aegis | 85.1 | 78.9 | 85.2 | 0% | 0.04 |
| Jev 1.14 | Gemini 3.8 Flash | civil | 79.7 | 92.1 | 92.1 | 94% | 0.96 |
| Jev 1.14 | Gemini 3.8 Flash | banking77 | 79.2 | 82.4 | 81.7 | 24% | 0.33 |
| Jev 1.14 | Gemini 3.8 Flash | arxiv | 89.4 | 91.2 | 89.5 | 11% | 0.15 |
| Jev 1.14 | Gemini 3.8 Flash | when2call | 74.2 | 74.8 | 76.5 | 8% | 0.13 |
| Jev 1.14 | Gemini 3.8 Flash | judgebench | 51.5 | 91.1 | 90.9 | 98% | 1.01 |
| Jev 1.14 | Gemini 3.8 Flash | helpsteer2 | 39.6 | 45.9 | 45.9 | 100% | 1.02 |

Cases where System Two is more accurate than System One: 99 of 168
Of these, the cascade keeps the accuracy (within 1 point) at < 50% of the cost: 28
