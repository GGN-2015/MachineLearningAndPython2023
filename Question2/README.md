# Binary Classification Questions

In this article, we will propose two binary classification problems related to music and some features that may be associated with these two problems. Based on my previous related practice with these two problems, we will analyze some features to illustrate the difficulties that these two binary classification problems may encounter in future workflow.

## Question1: Given an audio clip, analyze whether it belongs to the interlude of a song

Since there are many different types of music, we hereby declare that we do not study on the music called "pure music". In other words, we only focus on music that has "vocals". If in a piece of music, the lyrics are the main content expressed, we call this type of music "non-interlude music". Conversely, if the main content expressed in the music is various instruments, we call this type of music "interlude music".

For a music waveform, the first feature we obtain is a function of amplitude changing with time. Secondly, we can perform a short-time Fourier transform on this function of amplitude changing with time, to obtain a function of intensity changing with frequency (in other words, obtaining the frequency spectrum information of the music).

In this classification problem, based on our experience, we can obtain an attractive "classification shortcut" - that is, the loudness of music. Generally speaking, in the same piece of music, the loudness of the interlude part is often significantly lower than that of the main melody. 

However, this approach is not advisable. For example, when we study the probability of breast cancer, we may fall into an obvious shortcut: from the perspective of the samples, the probability of women getting breast cancer is often higher than that of men. However, our classifier is meant to classify issues related to breast cancer, not to classify the gender of samples. Therefore, I plan to measure the loudness of the songs first and divide the music segments into several categories based on different loudness levels. Then, I will further design classification algorithms within each category.

## Question2: Given a music segment with vocals, analyze whether the voice in this segment is "male," "female," or "child."

Similar to the previous question, frequency spectrum features are often important features in audio analysis. However, as we focus on issues related to vocals, the "Mel-frequency cepstral coefficients (MFCC)" may be more appropriate than commonly used frequency spectrum. Additionally, we may need to apply pre-emphasis to the frequency features to highlight the vocal parts.

Similar to the previous question, there is also a common shortcut in classifying human voices -- the fundamental frequency of the voice. In everyday communication, the frequency of male voices is often lower than that of female and child voices. However, this shortcut is not applicable in music, because we cannot tolerate classifying "tenor" as "female voice" or classifying "contralto" as "male voice".

## Furthermore

In my upcoming work, I will focus on researching issues related to Question 1. The singers of the music I study are virtual singers such as Luo Tianyi and Yan He (Since virtual singers may be a popular field in the future.), so the timbre of "human voice" is not the issue I will focus on researching. Due to my limited personal research experience in this area, I would greatly appreciate your valuable guidance.
