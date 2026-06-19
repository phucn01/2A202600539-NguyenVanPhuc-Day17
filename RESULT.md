# RESULT

## Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1300 | 13387 | 0.00 | 0.16 | 0 | 0 |
| Advanced | 2018 | 18342 | 0.64 | 0.79 | 6 | 0 |

## Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 237 | 22297 | 0.00 | 0.10 | 0 | 0 |
| Advanced | 318 | 12053 | 0.67 | 0.87 | 0 | 19 |

## Nhận Xét

- Advanced có recall tốt hơn vì lưu được fact ổn định vào `User.md`.
- Baseline gần như không tăng dung lượng memory vì không có persistent profile.
- Compact memory phát huy tác dụng rõ nhất trong bài stress dài vì nó giảm mạnh chi phí ngữ cảnh mà vẫn giữ được thông tin quan trọng.
