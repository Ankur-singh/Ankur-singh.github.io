## Steps to setup development environment
---

1. `sudo docker run -it --name jekyll_dev -v $(pwd):/srv/jekyll -p 4000:4000 jekyll/jekyll /bin/bash`
2. `cd <your project directory>`
3. `bundle install`
4. `bundle exec jekyll serve --host 0.0.0.0 --incremental`

Now, you can edit the code and see the changes in your browser.