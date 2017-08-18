---
layout: default
---

{% for post in site.posts %}	
## [{{ post.title }}]({{ post.url }})
{{ post.summary }}

- {{ post.date | date: "%B %e, %Y" }}			
{% endfor %}	