## Steps to setup development environment
---

1. Create docker container inside your project directory
```bash
sudo docker run -it --name jekyll_dev -v $(pwd):/srv/jekyll -p 4000:4000 jekyll/jekyll /bin/bash
```

2. Once you are inside the your docker container, run the following:
```bash 
cd /srv/jekyll
bunble install
bundle exec jekyll serve --host 0.0.0.0 --incremental
`

Now, you can edit the code and see the changes in your browser.
