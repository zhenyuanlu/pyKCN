
<p style="line-height:25px;margin:0px;"><br></p>

<p align="center">
  <img align='center' src="logo.png" width="70%" alt="pyKCN logo"/>
</p>

<p style="line-height:25px;margin:0px;"><br></p>

<p align="center">
    <b>pyKCN: A <em>Python</em> Tool for Bridging Scientific Knowledge through Keyword Analysis</b>
</p>

<p align="center">
<a href="https://img.shields.io/badge/version-0.1.0-blue" target="_blank">
    <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version">
</a>
<!--
<a href="https://pypi.org/project/pykcn" target="_blank">
    <img src="https://img.shields.io/pypi/v/pykcn?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/pykcn" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/pykcn.svg?color=%2334D058" alt="Supported Python versions">
</a>
-->

---

<br/>

[pyKCN: A <em>Python</em> Tool for Bridging Scientific](https://arxiv.org/abs/2403.16157)\
<b>Zhenyuan Lu</b>, Wei Li, Burcu Ozek, Haozhou Zhou, Srinivasan Radhakrishnan, Sagar Kamarthi

<br/>

Our team has previously published a series of related papers that laid the groundwork for the development of this tool. Here are those publications:

- [Novel keyword co-occurrence network-based methods to foster systematic reviews of scientific literature](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0172778) ![Novel keyword co-occurrence 
  network-based methods to foster systematic reviews of scientific literature Citations](https://img.shields.io/badge/Citations-0-blue) \
  Srinivasan Radhakrishnan, Serkan Erbis, Jacqueline A. Isaacs, Sagar Kamarthi
- [Analysis of pain research literature through keyword Co-occurrence network](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000331) ![Analysis of pain research literature through keyword Co-occurrence networks Citations](https://img.shields.io/badge/Citations-0-blue) \
  Burcu Ozek, **Zhenyuan Lu**, Fatemeh Pouromran, Srinivasan Radhakrishnan, Sagar Kamarthi
- [Trends in adopting industry 4.0 for asset life cycle management for sustainability: a keyword Co-occurrence 
  network review and analysis](https://www.mdpi.com/2071-1050/14/19/12233) ![Trends in adopting industry 4.0 for 
  asset life cycle management for sustainability: a keyword Co-occurrence network review and analysis Citations](https://img.shields.io/badge/Citations-0-blue) \
  Sachini Weerasekara, **Zhenyuan Lu**, Burcu Ozek, Jacqueline Isaacs, Sagar Kamarthi
- [Navigating the Evolution of Digital Twins Research through Keyword Co-Occurence Network Analysis](https://www.mdpi.com/1424-8220/24/4/1202) ![Navigating the Evolution of Digital Twins Research through Keyword Co-Occurence Network Analysis Citations](https://img.shields.io/badge/Citations-1-blue) \
  Wei Li, Haozhou Zhou, **Zhenyuan Lu**, Sagar Kamarthi

<br/>


## Abstract




pyKCN, a Python-based tool for analyzing co-occurrence keywords in literature review. pyKCN is a python tool that can be used to analyze the trending of a field through a robust analysis of co-occurrence keywords, association rules and other models. The tool is equipped with a comprehensive extractor module alongside a text processor, a deduplication processor, and several keyword analysis methods including KCN and association rule. The strength of pyKCN extends beyond literature analysis. It has been instrumental in propelling multiple studies across diverse domains, such as nano EHS, industry 4.0, pain research, etc. Furthermore, pyKCN's architecture enhance it with the ability to process and analyze large scale dataset, thereby providing a platform for researchers to visualize the important role of keywords within and across academic papers. This, in turn, empowers scholars to discern emerging trends, identify seminal works, and cultivate a nuanced understanding of the thematic and structural contours of scientific discourse. 


## Get Started


### Installation
This project requires Python 3.8 or newer.
```
biopython==1.83
nltk==3.8.1
numpy==1.26.4
pandas==2.1.1
rapidfuzz==3.6.1
xlrd==2.0.1
pyarrow==15.0.0 (optional)
```


## Reference

If you find our study useful, please cite our paper on arxiv:
```
@article{lu2024pykcn,
  title={pyKCN: A Python Tool for Bridging Scientific Knowledge},
  author={Lu, Zhenyuan and Li, Wei and Ozek, Burcu and Zhou, Haozhou and Radhakrishnan, Srinivasan and Kamarthi, Sagar},
  journal={arXiv preprint arXiv:2403.16157},
  year={2024}
}
```

## Author
[Zhenyuan Lu](https://zhenyuanlu.com/)\
Email: lu.zhenyua[at]northeastern[dot]edu

## License

This project is licensed under the terms of the [MIT license](LICENSE).
