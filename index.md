---
layout: default
---

{% for post in site.posts %}	
## [{{ post.title }}]({{ post.url }})
- {{ post.date | date: "%B %e, %Y" }}			
{% endfor %}	