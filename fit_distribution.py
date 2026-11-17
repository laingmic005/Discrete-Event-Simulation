'''
This script generates a best distribution expression for
Rockwell Automation: Arena based on sample data. The script
performs a Kolmogorov-Smirnov (KS) goodness-of-fit test to
compare each fitted distribution against the sample data.

Written by Isabell Hudnall on 11/17/2026
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# commonly used distributions
DISTRIBUTIONS = {
    "NORM": stats.norm,
    "LOGN": stats.lognorm,
    "EXPO": stats.expon,
    "WEIB": stats.weibull_min,
    "GAMM": stats.gamma,
    "UNIF": stats.uniform
}

def load_data(filename):
    with open(filename, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    return np.array(data)

def fit_distributions(data):
    results = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(data)

            ks_stat, p_value = stats.kstest(
                data,
                dist.name,
                args = params
            )

            results.append({
                "name": name,
                "params": params,
                "ks": ks_stat,
                "p": p_value,
                "dist": dist
            })

        except Exception as e:
            print(f"Failed {name}: {e}")

    results.sort(key=lambda x: x["ks"])
    return results

def arena_expression(name, params):
    if name == "NORM":
        return f"NORM({params[0]:.4f}, {params[1]:.4f})"

    elif name == "LOGN":
        shape, loc, scale = params
        return f"LOGN({scale:.4f}, {shape:.4f})"

    elif name == "EXPO":
        loc, scale = params
        return f"EXPO({scale:.4f}, {shape:.4f})"

    elif name == "GAMM":
        shape, loc, scale = params
        return f"GAMM({shape:.4f}, {scale:.4f})"

    elif name == "UNIF":
        loc, scale = params
        return f"UNIF({loc:.4f}, {loc+scale:.4f})"

    return str(params)

def plot_fit(data, best):
    dist = best["dist"]
    params = best["params"]

    x = np.linspace(min(data), max(data), 1000)
    pdf = dist.pdf(x, *params)

    plt.figure(figsize=(10,6))
    plt.hist(data, bins='auto', density=True, alpha=0.6, label='Data')
    plt.plot(x, pdf, 'r-', lw=2, label=best["name"])
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Best Fitted Distribution")
    plt.legend()
    plt.show()

def main():
    filename = 'Numberofindividualpartsinorder.txt'

    data = load_data(filename)
    results = fit_distributions(data)
    best = results[0]

    print("\nBest Distribution")
    print("------------------------")
    print(f"Name: {best['name']}")
    print(f"KS Statistic: {best['ks']:.6f}")
    print(f"p-Value: {best['p']:.6f}")

    print("\nArena Expression:")
    print(arena_expression(best["name"], best["params"]))

    plot_fit(data, best)

if __name__ == "__main__":
    main()

