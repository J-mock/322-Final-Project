# TODO : Implement same testing for previous assignments but with NBA related data
import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier

# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],     # False
    ["Senior", "Java", "no", "yes"],    # False
    ["Mid", "Python", "no", "no"],      # True
    ["Junior", "Python", "no", "no"],   # True
    ["Junior", "R", "yes", "no"],       # True
    ["Junior", "R", "yes", "yes"],      # False
    ["Mid", "R", "yes", "yes"],         # True
    ["Senior", "Python", "no", "no"],   # False
    ["Senior", "R", "yes", "no"],       # True
    ["Junior", "Python", "yes", "no"],  # True
    ["Senior", "Python", "yes", "yes"], # True
    ["Mid", "Python", "no", "yes"],     # True
    ["Mid", "Java", "yes", "no"],       # True
    ["Junior", "Python", "no", "yes"]   # False
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

X_test = [["Junior", "Java", "yes", "no"],
     ["Junior", "Java", "yes", "yes"],
     ["Intern", "Java", "yes", "yes"]] 

def test_decision_tree_classifier_fit():
    myTDIDT = MyDecisionTreeClassifier()
    myTDIDT.fit(X_train_interview, y_train_interview)

    assert myTDIDT.tree ==  tree_interview # TODO: fix this

def test_decision_tree_classifier_predict():
    myTDIDT = MyDecisionTreeClassifier()
    myTDIDT.fit(X_train_interview, y_train_interview)
    actual = ["True", "False", None]
    predicted = myTDIDT.predict(X_test)
    assert actual == predicted # TODO: fix this