# Smooth Private Forest
Implementation of Smooth Private Forest, a differentially private decision forest designed to minimize the number of queries required and the sensitivity of those queries. Originally published in: <br>

_Fletcher, S., & Islam, M. Z. (2017). Differentially private random decision forests using smooth sensitivity. Expert Systems with Applications, 78, 16-31._

## BibTeX
```
@article{fletcher2017differentially,
  title={Differentially private random decision forests using smooth sensitivity},
  author={Fletcher, Sam and Islam, Md Zahidul},
  journal={Expert Systems with Applications},
  volume={78},
  pages={16--31},
  year={2017},
  publisher={Elsevier}
}
```

## Installation

Either download SmoothPrivateForest from the Weka package manager, or download the latest release from the "Releases" section on the sidebar of Github.

## Compilation / Development

Set up a project in your IDE of choice, including weka.jar as a compile-time library.

## Options

`-N <number of trees in forest>`
Number of trees in forest. (default 10)

`-D <number of display trees>`
Number of trees to display in the output. (default 3)

`-E <epsilon>`
The privacy budget (epsilon) for the exponential mechanism. (default 1.0)
 
`-P`
Whether or not to display flipped majorities, sensitivity information and true distributions in leaves. (default true)

`-C <classname>`
Specify the full class name of the classifier to compare with Smooth Private Forest.

`-S <num>`
Seed for random number generator. (default 1)
