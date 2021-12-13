
"""
note that you only need to download the tokenizer model once from spacy.
"""
# import spacy
# spacy.cli.download("it_core_news_sm")
# spacy.cli.download("de_core_news_sm")
# spacy.cli.download("en_core_web_sm")
# spacy.cli.download("fr_core_news_sm")
# spacy.cli.download("es_core_news_sm")
# spacy.cli.download("pt_core_news_sm")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from ariadne import  load_schema_from_path, make_executable_schema, graphql_sync
from ariadne.constants import PLAYGROUND_HTML
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.main import Constants
from resolvers.mutations import mutation
app = Flask(__name__)
CORS(app)

type_defs = load_schema_from_path("schema.graphql")
schema = make_executable_schema(
    type_defs, mutation
)


@app.route("/graphql", methods=["GET"])
def graphql_playground():
    return PLAYGROUND_HTML, 200


@app.route("/graphql", methods=["POST"])
def graphql_server():
    data = request.get_json()
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=True
    )
    status_code = 200 if success else 400
    return jsonify(result), status_code

# @app.route('/', methods=["GET", "POST"])
# def home():
#     return jsonify({
#     }), 200


if __name__ == '__main__':
    app.run(debug=Constants.DEBUG, port=Constants.PORT)
