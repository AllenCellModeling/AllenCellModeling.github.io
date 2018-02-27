# Source of [allencellmodeling.github.io](https://allencellmodeling.github.io)


To add and display a new notebook as a page on the site:

## Things that you do to the notebook
- Make sure you have the figures sized as you want them to appear
- Add 'raw nbconvert' cell at the start of the notebook declaring the layout as post and the title of the post. This YAML front matter will be used to render the notebook's information on the index page, e.g.

```
---
title: Publishing a Jupyter-notebook
summary: A short explanation to appear on the index page. 
---
```

- Title the `.ipnb` file in the format `YEAR-MONTH-DAY-title` so that it becomes, e.g., `2017-08-17-test_notebook.ipynb`
- Set the `$NOTEBOOKNAME` bash variable as in `NOTEBOOKNAME=2017-08-17-test_notebook.ipynb`

## Things you do to the repo
- Pull from master 
- Put the `.ipynb` file in the `_posts` directory
- Convert the notebook to a markdown file using 
```
jupyter nbconvert --to markdown --NbConvertApp.output_files_dir='../assets/nbfiles/{notebook_name}' --FilesWriter.relpath='({{ site.url }}/assets/nbfiles/' $NOTEBOOKNAME
```
- This creates a markdown file like `2017-07-17-test_notebook.md` and the related images in `../assets/nbfiles/2017-07-17-test_notebook_files`
- Move the `.ipynb` to `../assets/notebooks`

The conversion and moving processes are scriptable for future improvement. 
