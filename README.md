# GANime-FullBody

Full body Generation of anime girls
Development & Research logs

## Purpose

- Satisfy my desire
  there's Anime Face Gans but I want more waifu!

- Can be used as a placeholder sprite in game development
  I also develop some noob games as a hobby
  and I wanted to use some good sprite as a placeholder
  or as the real sprite for my game

# Brief introduction to Gans

Gans or Generative Advisoral Networks to be exact
it's a class of machine learning frameworks by [Ian Goodfellow - Wikipedia](https://en.wikipedia.org/wiki/Ian_Goodfellow)
since its publication, there's tons of research and Implementation after it
Gans Mainly Involved two main components
and I'll conceptually introduce them with an example from
A fake money maker and Police

#### Generator - A Fake Moneymaker

first, we have a Generator by the name you can already what's his job
Generator or Moneymaker will be trying to generate data
in this case, an image of Fake bills to fool the police

#### Discriminator - A Police

police or discriminator on the other hand will be given an information
that there's a fake bill and it need to detect them (work like a classification model)

### Get Good Together

as the Moneymaker trying to fool and got caught by police
it'll learn what style and shape, color, and tone to use to fool the police
and as the police saw more bill experience fakes and reals data
it'll learn to get better at detecting
these result in a training loop where the two of them get good together
to the point that the money maker or Generator to be exact
can completely fool even good police/discriminator

**Note**: Though there's a variety of Gans models but they're all based on
Generator and Discriminator improvement
in this project, we'll work on dcgans, wgan-gp, progans, stylegans    

# Datasets

before getting our hands dirty with coding and model structure
Like other ML models, we need good data Gans was no exception
The conditions that we need to pay attention to are

- Clear Background

- Same or Similar Art Style

- A considerably large amount of data - 10k or above

But from the available datasets, I've found on the internet
There's **NONE** that satisfies these conditions
Especially  Full Body Image & Clear Background

<u>So there's no other way except to scrape it myself</u>

## Data-Collection

### Requirement

Firstly I've thought about the required condition for the site that I'll scrape from

- Easy to filter image (image filtering exist & or image tags Exist)

- A considerably large amount of data - 10k or above

- Satisfy Datasets requirements

### Selection

I've begun to search for a site that satisfies my requirement and I've found

| Site / Req          | Filter    | Data  | Datasets - Req |
|:-------------------:|:---------:|:-----:|:--------------:|
| Gelbooru            | very easy | large | satisfy        |
| Getchu              | very hard | large | satisfy        |
| Fandom              | very hard | large | not - satisfy  |
| anime-char-database | easy      | small | not - satisfy  |

Based on this table we Have

- Gelbooru

- Getchu

Which requires more inspection, They have their pain in the ass

#### Gelbooru - is NSFW

Gelbooru is a hentai image (porn) site
which results in their

##### PROS:

- detailed and & precise tags

- easy to scrape

##### CONS:

- it's NSFW (nude / porn / etc...)

- not ensure similar art style

- contain some datasets that are out of the domain

#### Getchu - is a pain

Getchu is a Visual Novel game seller that shows the character info
which results in their

##### PROS:

- Safe Image
- ensure considerably similar art style
- reliable quality of data

##### CONS:

- Also, Contain Half-Body images
- All Japanese with no option for English so it's harder to figure out
- Website hosted in jp server and it's slow so it's hard to scrape from my place

#### Selection - Conclusion

I've Come to the conclusion that'll scrape Getchu because it's

- similar art style
- reliable quality of data

But at that time Getchu server is too slow if I access it from my places
so I need to go with Gelbooru instead

But the Problem is Gelbooru is NSFW which is unacceptable

##### Alternative

But I managed to find a Gelbooru alternative
by simply searching "Gelbooru Safeforworks"

###### Safebooru - Gelbooru but it's safe

Alternative site for Gelbooru with the safe structure images and tags
but it excludes all NSFW contents

After figuring out how their website filter works
I know it'll take some time to write my web-scraper

**<u>But why would I write it myself LOL</u>**

### Scraping

I began to search on GitHub if there's an image crawler existed for these sites

###### An Encounter of Fate

I've found a simple Unknow-project But EXACTLY matched my need
It's an image crawler for booru site **which allows you to use tags for filtering**
[GitHub - LittleJake/animate-image-crawler: A crawler for booru site. (gelbooru and safebooru, etc.)](https://github.com/LittleJake/animate-image-crawler)

So I've skipped the whole process of writing a web scraper
thank ~~**our code☭**~~ this random project

I've manually tried many tags combination for Safebooru and I've found
**with --tags='full_body solo standing 1girl white_background'**

For filtering, We can get 1girl standing with clear-background
So I used it with this image crawler and got my hand on 20k images

---

## Data Cleaning

Well on 20k datasets with an efficiency of 70% it worth cleaning
but at this rate, we would need to clean out 6k of data
which is a terrible job, But I've already filtered 3k of data out manually
before I find the efficiency of data so we need to figure our way out from there

### Data Analysis

based on the 2k of data I've filtered out we can categorize filtered the data to

- Chibi

- More than 1 character

- Too flashy background

- Weird viewpoint

From 3k of data, the Majority of them(1:4) fall into "Chibi" category

Chibi

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-05-04-17-31-34-8c7f0d377b5c1fb9697cdb9b26be8f40fee7fd41.jpg)

"More than 1 character" and so on by order
More Than 1

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-05-04-17-38-51-sample_fa92cf481785d32b12a56f6b8e7d395e7aeb3cdd.jpg)

From my Point of view
I've found that Chibi Images is quite repetitive
in other words, it's easy to spot on

so why don't I train the classification model to filter them out
since it'll be easier to just train classification model than classified the images myself

### Classifier

#### Sequential

I began with the Sequential model but well it's too long so I'll skip my effort on this part
just to let you know it didn't work out well enough

#### VGG16

so transfer learning is the solution since I just need a good classifier with low effort :P
I decided to go with VGG16 and fine-tune the last fourth layer

```python
base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_images[0].shape)
```

##### Data

I split data into two category [''chibi,'not_chibi']

**Chibi** -> Chibi images

**not_chibi** -> Cleaned images

Crop image before resize to maintain aspect-ratio

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-18-32-image.png)

```python
def square_pad(image):
    w,h,c = image.shape
    if w>=h:
        out = np.zeros([w,w,c])
        out[:,int((w-h)/2):int((w-h)/2)+h,:] = image
    else:
        out = np.zeros([h,h,c])
        out[int((h-w)/2):int((h-w)/2)+w,:,:] = image
    return outturn out
```

##### Train

```python
# Freeze convolution blocks
for layer in base_model.layers[:15]:
    layer.trainable = False
```

RESULT:

<img src="https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-29-09-21-53-image.png" title="" alt="" width="651">

98% Accuracy which is pretty good for just 12 epoch
(well it's transfer learning ofc it would be good :p)

##### Filtering

###### Chibi - classifier

then I use it to filter the datasets and split them into chibi and not_chibi folder
It's not that fast with 5. X images/sec but it's still better than doing it manually
On 16k images

**Chibi**
![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-29-09-26-26-image.png)

**Not Chibi**

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-29-09-25-51-image.png)

The Result is Pretty Satisfying with 2.4k Chibi was Filtered out with just 5X bad picking!!!
Imagine doing that manually Lol
<u>NOTE: This model also performed well
with out-of-domain data Ex. Random internet chibi/not chibi character images</u>

###### not_single Filter

After that, we've got our hands on 13.5 k images
but we still have something like this in it "not_single" images

Which is Unacceptable

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-22-50-image.png)

So I'll just use the same code to train classification on it

**Square Pad them**

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-21-34-image.png)

**Train**

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-24-24-image.png)

Evaluate

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-25-09-image.png)

The result is satisfying so we can use it to filter the images

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-28-57-image.png)

The result is "OKAY" But not on par with Chibi - Classifier model
I Assume that the cause comes from lower datasets (APX 100 ~  images) to train with

So it predicted some single images as not_single
![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-32-26-image.png)

But it isn't too bad since I've filtered almost all "Not_single" images automatically
900~ Images were filtered out

I've left with 12k images
![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-37-41-image.png)

But still, There would've surely had something that I should cleanout

###### Here's Hell - Manually clean them again

So I get back to reality and started cleaning the datasets

The main concern that I'll filter them out is

- [x] **Not to my liking, out of overall style** - Kemo, Pokemon, Weird things

- [x] **Weird Pose** - yoga pose or something like that

- [x] **Too Much Effect** - An Explosion Effect / Sparkle / or some sort

From 12k datasets | 2k~ was filtered out - > And I've Left with 10k Cleaned Datasets

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-04-30-00-52-03-image.png)

I Then uploaded it to Google-Drive which You can found it here

https://drive.google.com/drive/folders/1szQAsdtu9U9Dum0FPflC93R3D-w8qFCC?usp=sharing

And that's the End of my datasets, For now

# Trainning

## Trying out my data

Because I'm new to Gans, and also new to machine learning
I'll start from the well know Gans which I know didn't suitable for full body gans
but who cares :P

### DC Gans - trying on real data

I've trained DC Gans on anime face datasets before
and it worked quite well

So I'm interested in trying it with my real datasets
But since it has too much detail to capture
I Assumed that it'll be a mess, by far the result's a mess
I've Faced vanishing gradient and mode collapse on my model
and the result is the same noise over and over

### WGAN-GP - just DC Gans but Better

So I decided to change the loss function to Wasserstein loss
which improves the stability of Gans model basically (Wgans)
but in my senpai(senior) code he uses WGAN-GP
which also changed the weight clipping method
used on discriminator to Gradient Penalty
[L03 - Wasserstein GANs with Gradient Penalty - YouTube](https://www.youtube.com/watch?v=v6y5qQ0pcg4)

The weight clipping is limit the range of weight
and if it's over the length we'll just use the max value

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-05-07-07-44-44-1-4hWCkkakgFiq3NU7g9WDbQ.png)

Result in simpler distribution (row 1 images) 
Compare to (row 2 images) using Wgans-GP

I don't know much in-depth about them but by far, This is what I understand
The result improves quite a lot but by far it's not that good
with 100~ epochs on 128x128 - > 10k datasets
(i resized it from my original 258x258 datasets)

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/07-05-2022-08-07-47.png)

It captured the shape of a human
at this point, it proved that the quality of my datasets
Is decent enough for the model to form the right shape

The data problem is resolved,
though the Model didn't capture the important detail of the data
Such as Face & Accessory, So I need to figure my way out of this

### ProGans - scaling layers together!

My Approach Is progans since I hope scaling model as we scale images
As progans paper claims, it should capture resolution as
Generator and Discriminator(critic) scaled together
The Results show quite an improvement
![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/progan.png)

But as you can see it still didn't capture a detailed feature of a body
so we need a better way to improve the performance

### StyleGans - why use random noise, use Style!

I've Found StyleGans, one of the famous architecture for Gans
The Brief introduction is instead of using random noise
we map those noise to style and style transfer it with ADaIN to the generated image
but Gans Model takes a long time to train so I'll go with **64x64** because it take less time
if I managed to improve it I'll Scale it up

<u><strong>Result on 64x64 resolution - latent size 256</strong></u>

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-05-27-19-00-25-image.png)with the latent size of 256 results is pretty shitty lol

**<u>Result on 64x64 resolution - latent size 512</u>**

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/df6908a90d6cc41a05e441158de5d48267f46c9b.png)
latent size of 512 makes it work (it should've worked if I use 512 latent sizes lol)
**Tips: Big latent Size can make the model take much time to adapt**
but it's better than **NOT ENOUGH** latents size

## Clean Data - YES Again...

As you can see I think the model did quite a good job on my data
but there's something that i would like to call **"Dark Matter"** in the data

![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-05-27-18-59-05-image.png)![](https://raw.githubusercontent.com/HRNPH/GANime-FullBody/main/images/2022-05-27-18-59-15-image.png)
