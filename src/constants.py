from pandas import to_datetime

def get_eval_windows() -> list[tuple]:
    val_windows = [
        ("2024-03-01", "2024-04-01"),
        ("2024-04-01", "2024-05-01"),
        ("2024-05-01", "2024-06-01"),
        ("2024-06-01", "2024-07-01"),
        ("2024-07-01", "2024-08-01")
    ]

    for i, (start, end) in enumerate(val_windows):
        val_windows[i] = (to_datetime(start), to_datetime(end))

    return val_windows


TEST_TIME_LOWER_BOUND = to_datetime("2024-09-01")
EVAL_WINDOWS = get_eval_windows()
