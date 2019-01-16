![](Imagine.png)

# Flowpoints ML
Create deep learning models without all the typing and dimension mismatches! Follow [this link](https://mariusbrataas.github.io/flowpoints_ml/?p=load?0&x&i&3&0,0&64,96_3&fc1&0&2,8&44,0&1,96,64_2&fc1act1&31&9&44,11&0_9&fc2&0&16&88,0&1,64,64_16&fc2act&31&10&88,11&0_8&fc1act2&40&10&44,22&_10&fc3&0&18&132,22&1,64,10_18&fc3act&36&&132,33&) to play around with this graphical interface on your own :)

## Overview
This project is used to host a website in which users can quickly create drafts for deep learning models and have the equivalent plug-and-play code output immediately.

The code output to the user is written in python and utilises [PyTorch](https://pytorch.org/).

- [Overview](#overview)
	- [How I use these diagrams](#how-i-use-these-diagrams)
- [User guide](#user-guide)
	- [Building new models](#building-new-models)
	- [Editing parameters](#editing-parameters)
	- [Sharing](#sharing)
- [Contributing to this project](#contributing-to-this-project)
- [About](#about)
	- [Background](#background)
	- [Pure JSX flowcharting spin-off](#pure-jsx-flowcharting-spin-off)
- [Dependencies](#dependencies)
- [License](#license)

### How I use these diagrams
1. Start by adding an input block and edit it's parameters to get the correct number of dimensions and features.
2. Add a bunch of blocks, organise them a little bit and add all connections.
3. Start from the top and move downstream, editing block parameters to ensure their outputs match the desired dimensions.
4. Add names to all blocks (if needed).
5. Quickly review code.
6. Copy code, plug it into existing pipeline, lean back, and watch the magic.


## User guide
Or maybe just play around with it yourself? Should be fairly easy to get the hang of it!

### Building new models
In the lower right corner of the [website](https://mariusbrataas.github.io/flowpoints_ml/) there are three buttons. These can be used to add new flowpoints (blocks) to the model, show/hide code, and copy all code to clipboard.

To build a new model just click the blue + button. This will add a flowpoint. If this is the first flowpoint in the diagram it's type will automatically be set to "input". This flowpoint is not responsible for performing any calculations, but helps determine what input parameters the model's "forward"-function should expect.

Click the + button again, and a new flowpoint shows up. On each side of a flowpoint there are connection points marked by a ">" ("x" if the flowpoint is an input). A flowpoint can have as many inputs or outputs as you want. Click the output of the first flowpoint to the input of the second flowpoint to connect them.

Inside the flowpoint - at the top right - there's a dropdown button. Click this to access and modify more of the flowpoint's settings.

Here's and [example](https://mariusbrataas.github.io/flowpoints_ml/?p=load?0&x&i&2&0,0&1,64_2&fc1&0&1&44,0&1,64,32_1&act1&40&3&44,12&_3&fc2&0&4&88,0&1,32,10_4&act2&36&&88,12&) for you!

The green <> button can be used to show or hide the code for the current model. This code is updated live as you edit the model (I find that kind of satisfying to watch). The syntax highlighting here relies on [this awesome project](https://github.com/conorhastings/react-syntax-highlighter).

### Editing parameters
When clicking the dropdown button inside a flowpoint all it's parameters will be revealed. Some of these are set automatically to match the dimensions of the input, but most of these parameters can be edited freely. Please note that the parser for the code __does not check whether given parameters are valid__. It only ensures that the data type is correct.

### Sharing
Sharing of models can be accomplished by simply sharing the url of the document you're working on. I don't have a proper server, and thus no storage space. To make it possible to share models I've had javascript continuously update the url with a link that contains a somewhat dense text representation of the entire model.

When a user tries to open a link to a model this string representation is passed as a query which a piece of javascript then uses to rebuild the model locally in the user's browser.



## Contributing to this project
I'd love some help maintaining this project or adding more and better functionality!

The code is maintained in the master branch, while the website is hosted from the gh-pages branch.

As of right now the code is not properly commented, but I'll do that pretty soon.

Some ideas I'd like to implement:

- __Support for other ML frameworks__ (Keras, TF...). This can probably be implemented easily in the flowchart-to-code parser (in the src/flowparser directory).
- __Pre-processing.__ Maybe something utilising the [torch data loader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), with support for multiple workers and use of [transforms](https://pytorch.org/docs/stable/torchvision/transforms.html).
- __Python load model by link??__ Maybe not possible, but would be awesome!

Also, this project is structured in a somewhat okay manner, but I'm not entirely happy with it, and might change it soon.

```
flowpoints_ml
├── .gitattributes
├── .gitignore
├── LICENSE
├── package-lock.json
├── package.json
├── README.md
│
├── public
│   ├── index.html
│   └── manifest.json
│
└── src
   ├── App.css
   ├── App.js
   ├── App.test.js
   ├── index.css
   ├── index.js
   ├── serviceWorker.js
   │
   ├── flowparser
   │   ├── Codepaper.js
   │   ├── CommonTools.js
   │   ├── Constructor.js
   │   ├── Fit.js
   │   ├── FlowOrder.js
   │   ├── FlowParser.js
   │   ├── FlowStateNames.js
   │   ├── Forward.js
   │   ├── Initials.js
   │   ├── PlotHist.js
   │   ├── SaveLoad.js
   │   └── Step.js
   │
   ├── flowpoint
   │   ├── DrawPoints.js
   │   ├── Flowpoint.js
   │   └── FlowpointSettings.js
   │
   └── helpers
       ├── AppBottom.js
       ├── DrawConnections.js
       └── TopBar.js
```



## About
### Background
As sort of a weekend-project I built this for myself to use, but figured others might find it useful as well.

Disclaimer: Needed more than a weekend.

Whenever I create machine learning models I try to break up the model to it's individual parts, and then visualise how these nodes connect to each other and how information change shape along the way to allow for new representations.

Very easy to to in one's head. Not as easy to explain. Other ML-geeks usually get it when I point to code or math, but people without the same background tend to have a harder time understanding this magic when they ask to take a peek under the hood.

Often I would just - with a high level of abstraction in every way - draw the diagrams I imagine and explain those. These are usually way easier to understand than code and math.

That's the first reason I built this.

The second reason would be that this makes it very easy for me to quickly build new models and share them with others. Instead of writing a lot of similar code or rewriting existing code I just edit blocks in a diagram.

The third reason (which I only realised after using this for a while) is that this makes version control a whole lot easier. In a project folder I maintain a text file which holds links for all the models I've tried in that project. That turned out to be way easier than maintaining a folder with files for every model, or a single huge file which contains all of them.

### Pure JSX flowcharting spin-off
More like some kind of spin-off-inception.

In the beginning of building this I figured I'd just use some flowchart-diagramming-plotting-library, but I was amazed to find that the few libraries out there were either difficult to use, required loads of dependencies, did not run properly, poorly styled, or I had to pay to use them.

Thus I built my own little system to suit my requirements. The first draft for the flowchart (not the code parser, that took some tinkering) was finished in only a couple of hours, relying only on SVG paths and html draggables.

I then realised I had almost built a general flowcharting library :D\
So pretty soon I'll upload such a library (naming it flowpoints) and tweaking this one (flowpoints ml) to utilise that new library instead of keeping it's own not-so-generaly-applicable code.

Updates on this coming soon. Would love some help making that the best flowcharting library available!



## Dependencies
Current dependencies in this project are:

- [Material UI](https://material-ui.com/)
- [React JS](https://reactjs.org/)
- [react-copy-to-clipboard](https://github.com/nkbt/react-copy-to-clipboard)
- [react-draggable](https://github.com/mzabriskie/react-draggable)
- [react-icons](https://github.com/react-icons/react-icons)
- [react-syntax-highlighter](https://github.com/conorhastings/react-syntax-highlighter)



## License
These tools are open sourced software, [licensed as MIT](https://github.com/mariusbrataas/flowpoints_ml/blob/master/LICENSE)
