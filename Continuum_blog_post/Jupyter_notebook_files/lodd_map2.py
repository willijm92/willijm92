from __future__ import division
import os
import numpy as np
import pandas as pd

from bokeh.sampledata import us_states
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, TapTool, OpenURL, ColumnDataSource


us_states = us_states.data.copy()
del us_states["HI"]
del us_states["AK"]

state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]

data = pd.read_csv('study_data.csv')

output_file("lodd_map.html", title="LODD Map")
TOOLS="pan,box_zoom,wheel_zoom,reset,resize,save,hover,tap"
fig = figure(title="NIST LODD/LODI Studies", toolbar_location="left",
    plot_width=1100, plot_height=700,tools=TOOLS)

p1 = fig.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color="#884444", line_width=2)

source = ColumnDataSource({'studys': data['Study'],'report': data['Report']})
url = "http://dx.doi.org/10.6028/@report"
p2 = fig.circle(data['Longitude'],data['Latitude'],source=source,size=10, color="navy", alpha=0.5)
p2.select(dict(type=HoverTool)).tooltips = [('Study Title','@studys')]
p2.select(dict(type=TapTool)).callback = OpenURL(url=url)

show(fig)
