from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/placement_demo.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 1200
cgpa = np.clip(rng.normal(7.2, 1.05, n), 4.0, 10.0)
internships = rng.poisson(0.9, n).clip(0, 4)
projects = rng.poisson(2.2, n).clip(0, 8)
coding = rng.integers(25, 101, n)
communication = rng.integers(30, 101, n)
aptitude = rng.integers(25, 101, n)
backlogs = rng.poisson(0.45, n).clip(0, 4)
logit = (-7.0 + 0.62*cgpa + 0.42*internships + 0.20*projects +
         0.025*coding + 0.012*communication + 0.018*aptitude - 0.8*backlogs)
prob = 1/(1+np.exp(-logit))
placed = rng.binomial(1, prob)
df = pd.DataFrame({
    "cgpa": cgpa.round(2), "internships": internships, "projects": projects,
    "coding_score": coding, "communication_score": communication,
    "aptitude_score": aptitude, "backlogs": backlogs, "placed": placed
})
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} synthetic demo rows to {OUT}")
