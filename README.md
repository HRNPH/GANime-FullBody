# GANime-FullBody

Full body Generation of anime girls
Development & Research logs

# Datasets

Like other ML models, we need good data
The conditions that we need to pay attention to are

- Full Body Image

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

"More than 1 character" and so on by order

From my Point of view
I've found that Chibi Images is repetitive
in other words, it's easy to spot on

so why don't I train the classification model to filter them out
since it'll be easier to just train classification model than classified the images myself

### Classifier

#### Sequential

I began with Sequential model but well it's too long so I'll skip my effort on this part
just know it didn't work out well enough

#### VGG16

so transfer learning is the solution since I just need a good classifier with low effort :d
I decided to go with VGG16

```python
base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_images[0].shape)
```

##### Data

I split data into two category [''chibi,'not_chibi']

**Chibi** -> Chibi images

**not_chibi** -> Cleaned images

Crop image before resize to maintain aspect-ratio

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

<img src="file:///C:/Users/hirun/AppData/Roaming/marktext/images/2022-04-29-09-21-53-image.png" title="" alt="" width="651">

98% Accuracy which is pretty good

##### Filtering

then I use it to filter the datasets and split them into chibi and not chibi folder

**Chibi**
![](C:\Users\hirun\AppData\Roaming\marktext\images\2022-04-29-09-26-26-image.png)

**Not Chibi**

![](C:\Users\hirun\AppData\Roaming\marktext\images\2022-04-29-09-25-51-image.png)
