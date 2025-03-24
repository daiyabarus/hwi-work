class Diff:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post

    @property
    def delta(self):
        return round(self.post - self.pre, 2)

    @property
    def delta_percent(self):
        if self.pre == 0:
            if self.post != 0:
                return float(100)
            else:
                return 0
        return abs(round((self.delta / self.pre) * 100, 2))

    @property
    def flag5_inc(self):
        """
        Delta%>= 5 Delta > 0 Improve, ' Delta%>= 5 Delta < 0 Degrade, Maintain
        APPLY FOR KPI LIKE SUCCESS RATE
        """
        if self.delta > 0 and self.delta_percent > 10:
            return "Improve"
        elif self.delta < 0 and self.delta_percent > 10:
            return "Degrade"
        else:
            return "Maintain"

    @property
    def flag5_dcr(self):
        """
        Delta%>= 5 Delta < 0 Improve, ' Delta% >= 5 Delta > 0 Degrade, Maintain
        APPLY FOR KPI LIKE DROP
        """
        if self.delta < 0 and self.delta_percent > 10:
            return "Improve"
        elif self.delta > 0 and self.delta_percent > 10:
            return "Degrade"
        else:
            return "Maintain"

    @property
    def threshold_flag_inc(self):
        """
        use to calculate baseline Increase is BETTER

        post is Baseline
        """
        if self.post <= self.pre:
            return "Pass"
        else:
            return "Fail"

    @property
    def threshold_flag_dec(self):
        """
        use to calculate baseline Decrease is BETTER

        pre is Baseline
        """
        if self.post >= self.pre:
            return "Pass"
        else:
            return "Fail"
