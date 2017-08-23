# Source of [allencellmodeling.github.io](https://allencellmodeling.github.io)


To add and display a new notebook as a page on the site:

- make sure you have the figures sized as you want them to appear
- add 'raw nbconvert' cell at the start of the notebook declaring the layout as post and the title of the post, e.g.

```
  ---
  title: Publishing a Jupyter-notebook
  summary: A short explanation to appear on the index page. 
  ---
```
- title the notebook in the format `YEAR-MONTH-DAY-title` so that the file on disc is, e.g., `2017-08-17-test_notebook.ipynb`
- pull from master 
- put the .ipynb notebook in the `_posts` directory
- convert the notebook to a markdown file using `jupyter nbconvert --to markdown --NbConvertApp.output_files_dir='../assets/nbfiles/{notebook_name}' --FilesWriter.relpath='({{ site.url }}/assets/nbfiles/' $NOTEBOOKNAME`
	- this would create `2017-07-17-test_notebook.md` and `../assets/nbfiles/2017-07-17-test_notebook_files` in our example
-  move the .ipynb to ../assets/notebooks

The conversion and moving processes are scriptable for future improvement. 
