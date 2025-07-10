# Geosolver
How good are deep learning networks at playing [Geoguessr](https://www.geoguessr.com/)?
# About
There is a game called GeoGuessr that became pretty popular over the last decade and I really enjoy watching the championships where the best players try to guess the exact geographical location looking at the StreetView image from Google. I also enjoy training deep learning models 🙂 When these two interests came together in my mind, the idea of this project was born. 

To be honest, there is not that much of a motivation for this project, so I’m going to treat it as a fun, though very interesting pet project. On the other hand, the problem of geolocation based on an image has some useful applications. 

The goal of the project is to create a deep learning model that will be capable of predicting geographical location given an image of that place.

Creation of such a model from scratch would be a very expensive and complex task, therefore, I’m going to use all available pre-trained models and data that might be useful to achieve the goal.

One such model is “StreetCLIP”, a CLIP model fine-tuned for geolocation, which is already a decent model for geolocation. Will I be able to push it to perform even better?

Now, as the model choice is naturally done at this point, there are 2 things that need to be handled before fine-tuning:
1. Data
2. Optimization objective

The amount and quality of the data is definitely important, but I don’t expect any obstacles related to gathering more samples, suitable for fine-tuning.

Optimization objective is what I care about much more, as it is not an obvious decision at all. There is actually a lot of research related to training and fine-tuning for geolocation and I expect to see more in the near future. This project doesn’t present itself as a proper, serious academic research, so I want to waive any expectations for the reader as soon as possible. On the other hand, this project is not a regular “paper implementation”, as I truly strive to come up with something innovative, because I like the challenge. I will do my best to log my thoughts, reflections and findings as I make progress towards the goal of the project.

Getting back to the “Optimization objective” topic. According to my research, the vast majority of papers approach the task of geolocation as either classification or retrieval. The classification approach tasks the model to predict the location, given the image. As the number of possible exact locations is huge, researchers usually come up with some ways to aggregate locations into areas and predict those instead. I’m not going to dive deep into details of this approach, because I’m personally much more excited about the retrieval approach. Why? I don’t really have any satisfying justification, even for myself. To the best of my knowledge, there is no such a paper that would prove either of these two approaches to be superior. The reason why I’m more interested in retrieval approach is because I find it very elegant when model creates meaningful high-dimensional representations of geographical locations.