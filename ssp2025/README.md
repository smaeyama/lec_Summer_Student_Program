# ssp2025
国立大学法人　総合研究大学院大学　先端学術院　核融合科学コース  
2025年度 夏の体験入学  
[https://soken.nifs.ac.jp/archives/open_campus/open2025](https://soken.nifs.ac.jp/archives/open_campus/open2025)

体験課題：   
物理情報付きニューラルネットワークによる偏微分方程式の数値解法  
Numerical Solution of Partial Differential Equations Using Physics-Informed Neural Networks  

担当教員：   
前山 伸也（メタ階層ダイナミクスユニット）、堀 久美子（複合大域シミュレーションユニット）  
Shinya Maeyama (Meta-hierarchy Dynamics Unit), Kumiko Hori (Complex Global Simulation Unit)  

課題の概要：  
近年、AIや機械学習技術の目覚ましい進展により、複雑な連続関数を近似的に表現できるニューラルネットワークは、さまざまな分野での応用が模索されています。中でも、物理法則を損失関数に組み込む物理情報付きニューラルネットワーク（Physics-Informed Neural Networks; PINNs）は、核融合プラズマを含む多様な物理現象を記述する偏微分方程式の新たな数値解法として注目を集めています。本課題では、PINNによる数値解法の基本的な仕組みに触れ、いくつかの例題を通して、その特徴や応用可能性について体験的に理解することを目指します。  
In recent years, remarkable progress in AI and machine learning technologies has led to growing interest in applying neural networks, which have the capability to approximate complex functions, to a wide range of research fields. Among these approaches, Physics-Informed Neural Networks (PINNs), which incorporate physical laws into the loss function, have emerged as a new method for solving partial differential equations that describe various physical phenomena, including fusion plasmas. In this course, participants will explore the fundamental concepts behind numerical methods using PINNs and gain hands-on experience through selected example problems, aiming to develop an understanding of their characteristics and potential for physics applications.  

ディレクトリ構成：  
`01_pinn`：  
物理情報付きニューラルネットワーク（Physics-Informed Neural Networks; PINNs）による楕円型偏微分方程式の数値解法  
Numerical solution of an elliptic partial differential equation by using PINN  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smaeyama/lec_Summer_Student_Program/blob/ssp2025/ssp2025/01_pinn/PINN.ipynb)
2025.5.1 ほぼ整備完了。MLPでPoisson方程式を解く。  
2025.6.24 Flax NNX APIに対応。

`02_pinn_boundary_layer`：  
漸近接続理論を組み込んだPINNによる境界層問題の数値解法  
Numerical solution of a boundary layer problem by using assymptically-matched PINN  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smaeyama/lec_Summer_Student_Program/blob/ssp2025/ssp2025/02_pinn_boundary_layer/PINN_boundary_layer.ipynb)
2025.5.1 コードは動くが、01_pinnとクラス構成等のコーディングが異なるので、見やすいように整備する必要あり。  
2025.6.24 Flax NNX APIに対応。クラス構成も整理。

`03_pino`：  
物理情報付き作用素学習 (Physics-informed Neural Operators; PINOs) による楕円型偏微分方程式のの汎用数値解法  
Numerical solution of a boundary layer problem by using PINO  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smaeyama/lec_Summer_Student_Program/blob/ssp2025/ssp2025/03_pino/PINO.ipynb)
2025.5.1 ほぼ整備完了。01_pinnに比べて、(i) DeepONet構造を取り入れた, (ii) 複数セットの問題条件について同時に解くように構成（解作用素の推定）することで未知の問題についても推定可能。  
2025.6.24 Flax NNX APIに対応。


