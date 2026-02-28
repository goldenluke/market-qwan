from arch import arch_model
import numpy as np

class MarkovGARCH:

    def __init__(self, returns):
        self.returns = returns

    def forecast_vol(self):

        model = arch_model(
            self.returns,
            vol="Garch",
            p=1,
            q=1,
            dist="t"
        )

        res = model.fit(disp="off")

        vol = res.conditional_volatility

        return np.array(vol)