+++
date = '{{ .Date }}'
draft = true
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
description = ''
tags = []
slug = '{{ .File.ContentBaseName }}'
+++
