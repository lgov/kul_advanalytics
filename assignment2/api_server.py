import getopt
import json
import logging
import sys

from flask import Flask, request

app = Flask(__name__)

def run_flask_server(port):
    # Listen on all interfaces on the specified port nr
    app.run(debug=True, host='0.0.0.0', port=port)

    pass


@app.route('/classify', methods=['POST'])
def post_image():
    record = json.loads(request.data)
    print(record)
    return json.dumps({'tag':'Spaghetti'})

def start(argv):
    try:
        opts, args = getopt.getopt(argv, shortopts=[], longopts=["port="])
    except getopt.GetoptError:
        logging.exception("Problem parsing the command line arguments.")
        sys.exit(2)

    port = 20000
    for opt, arg in opts:
        if opt == '--port':
            port = arg

    run_flask_server(port)

if __name__ == '__main__':
    start(sys.argv[1:])
