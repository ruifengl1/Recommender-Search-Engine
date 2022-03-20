'''
For development, Launch with gunicorn --threads 1 -b 0.0.0.0:80 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc
'''

from re import split
from flask import Flask, render_template
from doc2vec import *
import sys

# constants
TOP_RECORDS = 5

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles"""
    titles = [elem[1] for elem in articles]
    filenames = ['/article/'+ elem[0] for elem in articles]
    zipped = zip(filenames, titles)
    return render_template('articles.html', zipped = zipped)


@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    filedir = f'{topic}/{filename}'
    for record in articles:
        if record[0] == filedir:
            result = record
            break
    title = result[1]
    content = result[2].splitlines()
    recommend_articles = recommended(result, articles, TOP_RECORDS)
    files = [('/article/'+ elem[1][0], elem[1][1]) for elem in recommend_articles]
    return render_template('article.html', title = title, content = content, recommend_articles = files)
    

# initialization
i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
articles = load_articles(articles_dirname, gloves)

# if __name__ == '__main__':
#     # initialization
#     glove_filename = './glove.6B.300d.txt'
#     articles_dirname = './bbc'

#     gloves = load_glove(glove_filename)
#     articles = load_articles(articles_dirname, gloves)
#     app.run(host='0.0.0.0', port=80)


