# GramerIt

## Description

This project is a clone of the popular grammar and spell-checking tool, Gramerly. It aims to provide users with a similar experience by offering grammar and spelling suggestions for their text.

## Features

- Grammar and spelling suggestions
- Text analysis and error detection

## Technologies Used

- Programming Language: Python
- Libraries: Numpy, Tensorflow, Logging, CLI

## Installation

1. Clone the repository: `git clone [repository URL]`
2. Install dependencies: `pip install -r requirements.txt`
3. Load the model into the weight folder
    ```
    cd weights
    wget 'https://storage.cloud.google.com/jayant07-ml-weights/output/bigramprobab2.json'   # For basic probability model

    ```

## Usage

1. Check the quality of text
    `python src/main.py --help`


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Contact

For any questions or inquiries, please contact Jayant Aanand at jayantanand2001@gmail.com .


### Contribute
1. Directory structure
```
```
2. 

### Performance output
```
{'basic_markov_chain': {'ai_generated': {'score': -9.705957370590738, 'time_taken': 0.0010056495666503906
        }, 'random': {'score': -13.337060757532837, 'time_taken': 0.00395965576171875
        }, 'human': {'score': -9.89921650595109, 'time_taken': 0.003034830093383789
        }
    }, 'linear_regression': {'ai_generated': {'score': -12.891879937482976, 'time_taken': 0.359882116317749
        }, 'random': {'score': -13.088416665127028, 'time_taken': 0.6559979915618896
        }, 'human': {'score': -12.94121817337031, 'time_taken': 0.6620304584503174
        }
    }, 'deep_learning': {'ai_generated': {'score': array([
                [
                    -11.187444
                ]
            ]), 'time_taken': 31.866721153259277
        }, 'random': {'score': array([
                [
                    -12.784488
                ]
            ]), 'time_taken': 58.870445728302
        }, 'human': {'score': array([
                [
                    -11.1594715
                ]
            ]), 'time_taken': 57.691577434539795
        }
    }
}
```