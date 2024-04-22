def get_ema_names(span):
    return f"EMA_{span}"


def generate_pinescript(entry_dates, exit_dates):
    str_before = """
//@version=5
indicator("Manual Trades", overlay=true)

var int[] entryDates = array.new_int()
var int[] exitDates = array.new_int()
"""

    template_entry_dates = "array.push(entryDates, {ts})"
    template_exit_dates = "array.push(exitDates, {ts})"

    str_after = """
for i = 0 to array.size(exitDates) - 1
    label.new(x=array.get(entryDates, i), xloc=xloc.bar_time, y=close, yloc=yloc.belowbar, color=color.green, textcolor=color.white, style=label.style_label_up)
    label.new(x=array.get(exitDates, i), xloc=xloc.bar_time, y=close, yloc=yloc.abovebar, color=color.red, textcolor=color.white, style=label.style_label_down)
    """

    entry_dates_str = "\n".join(
        [template_entry_dates.format(ts=entry_date) for entry_date in entry_dates]
    )
    exit_dates_str = "\n".join(
        [template_exit_dates.format(ts=exit_date) for exit_date in exit_dates]
    )

    return f"""{str_before}
{entry_dates_str}
{exit_dates_str}
{str_after}
           """


entry_dates = [1513555199999, 1517788799999]
exit_dates = [1519603199999, 1520812799999]
print(generate_pinescript(entry_dates, exit_dates))
