### Language Translation and Identification

this machine/deep learning api that will be served as a `graphql-api` using ariadne, to perform the following tasks.

### 1. Language Identification

<p align="center" ><img alt="" width="100%" src="/translate.gif"/></p>

Identifying the language which the text belongs to using a simple text classification model. This model will be able to identify 7 different languages:

1. english (en)
2. french (fr)
3. german (de)
4. spanish (es)
5. italian (it)
6. portuguese (pt)
7. swedish (sw)

### 2. Language Translation

<p align="center" ><img alt="" width="100%" src="/ident.png"/></p>
Language translation offers a bi-direction english to another language translation for example `english-to-french`. The model translation api will be able to translate the following languages:

1. eng-de (english to german)
2. de-eng (german to english)
3. eng-af (english to afrikaans)
4. af-eng (afrikaans to german)
5. fr-eng (french to german)
6. eng-fr (english to french)
7. es-eng (spanish to german)
8. eng-es (english to spanish)
9. it-eng (italian to german)
10. eng-it (english to italian)
11. pt-eng (portuguese to german)
12. eng-pt (english to portuguese)
13. sw-eng (swedish to german)
14. eng-sw (english to swedish)

### Starting the server

To start the server first you need to install all the packages that we used and make sure you have the `.pt` files for both the translation and identification models. To install the packages you need to run the following command:

> _Note that to save the `.pt` files for model you have to train the models first. The notebooks for doing so can be found on the repositories links that are given at the end of this README file._

```shell
pip install -r requirements.txt
```

### Models Metrics Summary

1. Language Translation models

 <table border="1">
    <thead>
      <tr>
        <th>model name</th>
        <th>model description</th>
        <th>BLEU metric</th>
        <th>test PPL</th>
        <th>challenges</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>eng-de</td>
        <td>translate sentences from english to germany.</td>
        <td>36.64</td>
        <td>8.807</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>de-eng</td>
        <td>translate sentences from germany to english.</td>
        <td>46.20</td>
        <td>7.783</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
         <tr>
        <td>eng-af</td>
        <td>translate sentences from english to afrikaans.</td>
        <td>0.00</td>
        <td>23.635</td>
        <td>the dataset that i used was having few examples.</td>
      </tr>
      </tr>
         <tr>
        <td>eng-af</td>
        <td>translate sentences from english to afrikaans.</td>
        <td>0.00</td>
        <td>23.635</td>
        <td>the dataset that i used was having few examples.</td>
      </tr>
      <tr>
        <td>es-eng</td>
        <td>translate sentences from spanish to english.</td>
        <td>44.12</td>
        <td>8.097</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>eng-es</td>
        <td>translate sentences from english to spanish.</td>
        <td>33.74</td>
        <td>12.877</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
      <tr>
        <td>eng-fr</td>
        <td>translate sentences from english to french.</td>
        <td>52.45</td>
        <td>8.803</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>fr-eng</td>
        <td>translate sentences from french to english.</td>
        <td>40.17</td>
        <td>8.803</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>eng-it</td>
        <td>translate sentences from english to italian.</td>
        <td>48.90</td>
        <td>6.288</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>it-eng</td>
        <td>translate sentences from italian to english.</td>
        <td>72.67</td>
        <td>2.530</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>eng-pt</td>
        <td>translate sentences from portuguese to french.</td>
        <td>45.92</td>
        <td>7.721</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>pt-eng</td>
        <td>translate sentences from portuguese to english.</td>
        <td>58.23</td>
        <td>4.371</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
       </tr>
        <tr>
        <td>eng-sw</td>
        <td>translate sentences from swedish to french.</td>
        <td>26.19</td>
        <td>11.406</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
        <tr>
        <td>sw-eng</td>
        <td>translate sentences from swedish to english.</td>
        <td>37.13</td>
        <td>10.160</td>
        <td>the model trains for a short period of time due to google colab session limitations.</td>
      </tr>
    </tbody>
  </table>

2. Language Identification models

For language identification i used the model based on [fasttext paper](https://arxiv.org/abs/1607.01759) for quick training on google colab `GPU`

<table border="1">
    <thead>
      <tr>
        <th>model name</th>
        <th>model description</th>
        <th>test accuracy</th>
        <th>validation accuracy</th>
        <th>train accuracy</th>
         <th>test loss</th>
        <th>validation loss</th>
        <th>train loss</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>best-lang-ident-model</td>
        <td>identifies which language does the sentence belongs to.</td>
        <td>99.22%</td>
        <td>99.00%</td>
        <td>100%</td>
        <td>0.036</td>
        <td>0.036</td>
        <td>0.000</td>
      </tr>
       </tbody>
  </table>

### Language Translation Model (graphql api)

The graphql server is running on `http://127.0.0.1:3002/graphql` if you send the following graphql mutation:

```
mutation Translator($input: TranslationInputType!) {
  translate(input: $input) {
    from_
    meta {
      name
      language
      author
      package
      description
      project
    }
    translation
    sent
  }
}
```

With the following `query` variables:

```json
{
  "input": {
    "to": "eng",
    "from_": "it",
    "text": "ciao , come stai ?"
  }
}
```

You will get the following response:

```json
{
  "data": {
    "translate": {
      "from_": "it",
      "meta": {
        "author": "@crispengari",
        "description": "language identification and translation graphql api.",
        "language": "python",
        "name": "ml backend",
        "package": "pytorch",
        "project": "noteme"
      },
      "sent": "ciao , come stai ?",
      "translation": "hello , how are you ? ."
    }
  }
}
```

### Language Identification Model (graphql api)

To identify the language that the text is written in, we run the following mutation on `http://127.0.0.1:3002/graphql`

```
mutation Identify($input: IdentificationInputType!) {
  identify(input: $input) {
    probability
    label
    lang
    prediction {
      code
      id
      name
    }
    predictions {
      prediction {
        code
        id
        name
      }
      probability
    }
  }
}
```

With the following query variables:

```json
{
  "input": {
    "text": "how are you?"
  }
}
```

To get the following response:

```json
{
  "data": {
    "identify": {
      "label": 0,
      "lang": "eng",
      "prediction": {
        "code": "eng",
        "id": 0,
        "name": "english"
      },
      "predictions": [
        {
          "prediction": {
            "code": "eng",
            "id": 0,
            "name": "english"
          },
          "probability": 1
        },
        {
          "prediction": {
            "code": "swe",
            "id": 1,
            "name": "swedish"
          },
          "probability": 0
        },
        {
          "prediction": {
            "code": "fra",
            "id": 2,
            "name": "french"
          },
          "probability": 0
        },
        {
          "prediction": {
            "code": "deu",
            "id": 3,
            "name": "germany"
          },
          "probability": 0
        },
        {
          "prediction": {
            "code": "ita",
            "id": 4,
            "name": "italian"
          },
          "probability": 0
        },
        {
          "prediction": {
            "code": "por",
            "id": 5,
            "name": "portuguese"
          },
          "probability": 0
        },
        {
          "prediction": {
            "code": "afr",
            "id": 6,
            "name": "afrikaans"
          },
          "probability": 0
        }
      ],
      "probability": 1
    }
  }
}
```

### Why graphql?

With graphql we allow the `client` to select `fields` he/she is interested in. And this give us an advantage of using a single endpoint for example `http://127.0.0.1:3002/graphql` for all the identification and translation models.

### Why language translation?

This project was build to translate simple and complex sentences for 7 different languages. The idea was brought forward with the project `likeme` where we perform some processing on user's caption using pytorch deep learning models. The following steps were considered to preprocess the caption:

1. identify the language the caption in
2. translate the given caption to a certain language.

### Notebooks

1. Translation models

- All the notebooks for the translation models are found [here](https://github.com/CrispenGari/nlp-pytorch/tree/main/07_NMT_Project)

2. Identification model

- The notebook for language identification model is found [here](https://github.com/CrispenGari/nlp-pytorch/tree/main/06_Language_Identification)
