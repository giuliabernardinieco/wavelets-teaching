def plot_wavelet_power(power_df, power_type, title="my_title", directory="."):
    
    '''Plot wavelet power as a function of time and period, with a secondary y-axis showing period in days. 
    Also includes a panel showing the time-averaged power as a function of period.
    
    Parameters:
    power_df: pandas DataFrame containing the wavelet power data. Must include columns 'period', 'time', and either 'xypower_rectified' (for crosspower) or 'rectified_power' (for wavelet power).
    power_type: string, either 'crosspower' or 'wavelet_power'.
    title: string, title for the plot.
    '''

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator
    from matplotlib.gridspec import GridSpec

    if power_type == "crosspower":
        grid = power_df.pivot(index="period", columns="time", values="xypower_rectified")
        avg_power = power_df.groupby("period")["xypower_rectified"].mean()
        cmap = "plasma"
        label = "Rectified Crosspower"

    elif power_type == "wavelet_power":
        grid = power_df.pivot(index="period", columns="time", values="rectified_power")
        avg_power = power_df.groupby("period")["rectified_power"].mean()
        cmap = "viridis"
        label = "Rectified Wavelet Power"

    else:
        raise ValueError("power_type must be 'crosspower' or 'wavelet_power'")

    period = grid.index.values
    time = grid.columns.values
    power = grid.values

    # --- Figure layout ---
    fig = plt.figure(figsize=(15,4))
    gs = GridSpec(1, 2, width_ratios=[1,4], wspace=0.05)

    ax_avg = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharey=ax_avg)

    # --- Left panel: time-averaged power ---
    ax_avg.plot(avg_power.values, period, color='black')
    ax_avg.set_xscale("log")
    ax_avg.set_yscale("log")
    ax_avg.set_xlabel("Time Avg Power")
    ax_avg.set_ylabel("Period")
    ax_avg.set_ylim(32, max(period))

    # --- Main wavelet power panel ---
    pcm = ax_main.pcolormesh(
        time,
        period,
        power,
        shading="auto",
        norm=LogNorm(),
        cmap=cmap
    )

    ax_main.set_yscale("log")
    ax_main.set_xlabel("Time")
    ax_main.set_ylim(32, max(period))
    ax_main.set_title(f"{title} - {label}")

    # hide duplicate y ticks
    plt.setp(ax_main.get_yticklabels(), visible=False)

    # --- Colorbar ---
    cbar = fig.colorbar(pcm, ax=ax_main, pad=0.1, label=label, shrink=0.7)

    # --- Secondary axis ---
    secax = ax_main.secondary_yaxis("right")
    day_ticks = [182, 365, 730, 1825, 3650, 5475]

    secax.yaxis.set_major_locator(FixedLocator(day_ticks))
    secax.set_yticklabels([f"{d} d" for d in day_ticks])
    secax.set_ylabel("Period (days)")
    secax.minorticks_off()
    
    # save the figure
    plt.savefig(f"{directory}/{title}_{power_type}.png", dpi=600, bbox_inches='tight')

# make similar plot for coherence, but with a different color map and label
def plot_coherence(coherence_df, title="my_title", directory = "."):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator
    from matplotlib.gridspec import GridSpec

    grid = coherence_df.pivot(index="period", columns="time", values="R2ns")
    avg_coherence = coherence_df.groupby("period")["R2ns"].mean()

    period = grid.index.values
    time = grid.columns.values
    coherence = grid.values

    # --- Figure layout ---
    fig = plt.figure(figsize=(15,4))
    gs = GridSpec(1, 2, width_ratios=[1,4], wspace=0.05)

    ax_avg = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharey=ax_avg)

    # --- Left panel: time-averaged coherence ---
    ax_avg.plot(avg_coherence.values, period, color='black')
    ax_avg.set_xscale("log")
    ax_avg.set_yscale("log")
    ax_avg.set_xlabel("Time Avg Coherence")
    ax_avg.set_ylabel("Period")
    ax_avg.set_ylim(32, max(period))

    # --- Main coherence panel ---
    pcm = ax_main.pcolormesh(
        time,
        period,
        coherence,
        shading="auto",
        cmap="Blues"
    )

    ax_main.set_yscale("log")
    ax_main.set_xlabel("Time")
    ax_main.set_ylim(32, max(period))
    ax_main.set_title(f"{title} - Coherence")

    # hide duplicate y ticks
    plt.setp(ax_main.get_yticklabels(), visible=False)
    ax_main.minorticks_off()

    # --- Colorbar ---
    cbar = fig.colorbar(pcm, ax=ax_main, pad=0.1, label="Coherence", shrink=0.7)
    
    # --- Secondary axis ---
    secax = ax_main.secondary_yaxis("right")
    day_ticks = [182, 365, 730, 1825, 3650, 5475]

    secax.yaxis.set_major_locator(FixedLocator(day_ticks))
    secax.set_yticklabels([f"{d} d" for d in day_ticks])
    secax.set_ylabel("Period (days)")
    secax.minorticks_off()
    
    # save the figure
    plt.savefig(f"{directory}/{title} - Coherence.png", dpi=600, bbox_inches='tight')

# plot the phase as a function of time and period, with a secondary y-axis showing period in days, no time averaged panel
def plot_phase(phase_df, title="my_title", directory="."):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator

    grid = phase_df.pivot(index="period", columns="time", values="phasexy")
    period = grid.index.values
    time = grid.columns.values
    phase = grid.values

    # --- Figure layout ---
    fig, ax_main = plt.subplots(figsize=(10,4))

    # --- Main phase panel ---
    pcm = ax_main.pcolormesh(
        time,
        period,
        phase,
        shading="auto",
        cmap="twilight"
    )

    ax_main.set_yscale("log")
    ax_main.set_xlabel("Time")
    ax_main.set_ylim(32, max(period))
    ax_main.set_title(f"{title} - Phase")
    ax_main.minorticks_off()

    # --- Colorbar ---
    cbar = fig.colorbar(pcm, ax=ax_main, pad = 0.15,label="Phase (degrees)", shrink=0.7)

    # --- Secondary axis ---
    secax = ax_main.secondary_yaxis("right")
    day_ticks = [182, 365, 730, 1825, 3650, 5475]

    secax.yaxis.set_major_locator(FixedLocator(day_ticks))
    secax.set_yticklabels([f"{d} d" for d in day_ticks])
    secax.set_ylabel("Period (days)")
    secax.minorticks_off()
    
        # save the figure
    plt.savefig(f"{directory}/{title}_Phase.png", dpi=600, bbox_inches='tight')
    