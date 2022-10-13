# Spotify Wrapped 🌈 Custom Stable Diffusion Video

This repo contains the work for the work I have done in preparation for the Spotify Wrapped 2022 event in Berlin.

In the event, a stage is setup with different props, lights, and funny elements. People can adjust the setting as they like, and choose between very classical hits like ```Yellow Submarine```, or modern songs like ```As it was```. When all is ready and the mood is set, a photographer takes a picture which is automatically uploaded to an S3 bucket, and sent as REST request alongside the chosen song to an endpoint exposed by the FastAPI server provided in this repo.

The server runs a custom made, optimized version of Stable Diffusion with Adabins and MiDaS to create a morphing 3D animation of the picture into a world of colors and shapes. For example:

https://user-images.githubusercontent.com/3061306/195661379-e276e441-9ee5-41ea-993e-d9cc05a23b83.mp4

The repo is based on the amazing work of [Deforum's Stable Diffusion](https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb). I optimized, cleaned, refactored, and organized the code, as well as extending it with the music video generation feature, preset system, REST server and more.

The repo is still WIP.

This readme will soon be updated with more information.
