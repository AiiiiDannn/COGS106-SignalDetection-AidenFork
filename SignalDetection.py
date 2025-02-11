import scipy.stats as stats  # Recommended from ChatGPT, spicy.stats provides faster computational speed, lower complexity and more accurate results.


class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        if (self.hits + self.misses) == 0:
            return 0.0
        return self.hits / (self.hits + self.misses)

    def false_alarm_rate(self):
        if (self.false_alarms + self.correct_rejections) == 0:
            return 0.0
        return self.false_alarms / (self.false_alarms + self.correct_rejections)

    def d_prime(self):
        H = self.hit_rate()
        FA = self.false_alarm_rate()

        return stats.norm.ppf(H) - stats.norm.ppf(FA)

    def criterion(self):
        H = self.hit_rate()
        FA = self.false_alarm_rate()

        return -0.5 * (stats.norm.ppf(H) + stats.norm.ppf(FA))


if __name__ == "__main__":
    sd = SignalDetection(
        hits=10, misses=20, false_alarms=15, correct_rejections=5
    )  # Just an exmaple valuees, match with my manual calculation.
    print("Hit rate:", sd.hit_rate())
    print("False alarm rate:", sd.false_alarm_rate())

    print("d' value:", sd.d_prime())
    print("Criterion value:", sd.criterion())
