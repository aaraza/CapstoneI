#This is the entry file, like index.html
#First we should import Flask
#Next we import render_template to manage templates (need a templates/ folder)
from __future__ import print_function
import flask
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
#from __future__ import print_function
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__) #__name__ = Placeholder for current module
Bootstrap(app)
colors = {
    'Black': '#000000',
    'Red':   '#FF0000',
    'Green': '#00FF00',
    'Blue':  '#0000FF',
}

def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]

#Use routes to define directories
@app.route('/')
def index():
	# Grab the inputs arguments from the URL
    	args = flask.request.args

    	# Get all the form arguments in the url with defaults
    	color = getitem(args, 'color', 'Black')
    	_from = int(getitem(args, '_from', 0))
    	to = int(getitem(args, 'to', 10))

    	# Create a polynomial line graph with those arguments
    	x = list(range(_from, to + 1))
    	fig = figure(title="Polynomial")
    	fig.line(x, [i ** 2 for i in x], color=colors[color], line_width=2)

    	js_resources = INLINE.render_js()
    	css_resources = INLINE.render_css()

    	script, div = components(fig)
    	html = flask.render_template(
        	'index.html',
        	plot_script=script,
        	plot_div=div,
        	js_resources=js_resources,
        	css_resources=css_resources,
        	color=color,
        	_from=_from,
        	to=to
    	)
    	return encode_utf8(html)
	# return 'Index' Normally don't return a string
	#return render_template('index.html') #Normally return a template

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/disclosure')
def disclosure():
	return render_template('disclosure.html')

#Run the app
if __name__ == '__main__':
	print(__doc__)
	app.run(debug=True) #Deubug is set to true