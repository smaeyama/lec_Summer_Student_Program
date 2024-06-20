# hweq2d
長谷川-若谷方程式に基づく抵抗性ドリフト波乱流の2次元数値シミュレーション

## 物理的問題設定

このプロジェクトは、長谷川-若谷方程式に基づいて抵抗性ドリフト波乱流を2次元でシミュレーションすることを目的としています。長谷川-若谷モデルは、以下のような方程式で表されます。

### 渦度方程式（Vorticity Equation）
$$ \frac{\partial \tilde{u}}{\partial t} + [\tilde{\phi}, \tilde{u}] = C(\tilde{\phi} - \tilde{n}) - \nu \nabla^4 \tilde{u} $$

### 連続の方程式（Continuity Equation）
$$ \frac{\partial \tilde{n}}{\partial t} + [\tilde{\phi}, \tilde{n}] + \kappa \frac{\partial \tilde{\phi}}{\partial y} = C(\tilde{\phi} - \tilde{n}) - \nu \nabla^4 \tilde{n} $$

ここで、
- $` \tilde{u} = \nabla^2 \tilde{\phi} `$（渦度）
- $` [\tilde{\phi}, \tilde{u}] = \mathbf{b} \times \nabla \tilde{\phi} \cdot \nabla \tilde{u} = \partial_x \tilde{\phi} \partial_y \tilde{u} - \partial_y \tilde{\phi} \partial_x \tilde{u} `$（ポアソン括弧）
- $` \kappa = -\frac{\partial \ln n_0}{\partial x} `$（逆密度勾配長）
- $` C = -\frac{T_e}{\eta e^2 n_0} \nabla_\parallel^2 = - c_a \partial_y^2 `$（断熱パラメータ）

※専門的な補足
1. 断熱パラメータ $`C`$ の扱いに多少歴史的経緯があって、若谷先生の原著論文 [M. Wakatani and A. Hasegawa, "A collisional drift wave description of plasma edge turbulence", Phys. Fluids 27, 611 (1984). http://dx.doi.org/10.1063/1.864660] では成長率が最大になるように $`\nabla_\parallel`$ を選ぶことにしていました。別の論文 [R. Numata, et al., "Bifurcation in electrostatic resistive drift wave turbulence", Phys. Plasmas 14, 102312 (2007). https://doi.org/10.1063/1.2796106] では磁気面上一様揺動 $`k_y=0`$ に対する電子応答の違いから断熱パラメータ $`C`$ を切り替えると帯状流が卓越するようになる修正長谷川-若谷モデルが議論されました。本コードでは、parameters.namelistファイルで `flag_adiabaticity = "kysquare"` とパラメータ設定すると、スラブ配位を想定して $`C \sim k_z^2 \sim k_y^2`$ としたモデルになるようにしています。
2. 超粘性項（$`\nu \nabla^4`$）のモデルは論文によって異なるが、ここではあまり効かないパラメータ領域を想定します。


## 使用方法

1. **プログラムのコンパイル**
    ターミナル上でメイクコマンドによりコンパイルします。
    ```bash
    make
    ```

2. **シミュレーションパラメータ設定**
    ```param.namelist``` ファイルで格子点数や時間刻み幅、プラズマパラメータを設定します。

3. **シミュレーションコードの実行**
    コンパイル後、以下のコマンドで実行します。
    ```bash
    ./hweq2d.exe
    ```

    数値シミュレーション結果は以下のNetCDFファイルに出力されます。
    ```bash
    data/phiinxy_t00000000.nc
    data/phiinxy_t00000001.nc
    ...
    ```
    時刻ステップ it = 0, 1, ... での $` \tilde{\phi}, \tilde{n}, \tilde{u}`$ の2次元(x,y)分布データ。
    ```bash
    data/phiinkxky_t00000000.nc
    data/phiinkxky_t00000001.nc
    ...
    ```
    時刻ステップ it = 0, 1, ... でのFourier係数 $` \tilde{\phi}_{k_x,k_y}, \tilde{n}_{k_x,k_y}, \tilde{u}_{k_x,k_y}`$ の2次元波数空間(kx,ky)分布データ。  

4. **ポスト処理データ解析**
    シミュレーションデータのNetCDFファイルを読み込んで図示したり、データ分析を行う Python スクリプトを diag/ ディレクトリに置いてあります。
    ```
    animation_phiinxy.ipynb   # phi(x,y) の2次元分布アニメーション
    animation_phiinkxky.ipynb # |phi(kx,ky)|^2 の2次元分布アニメーション
    plot_phiintky.ipynb       # エネルギースペクトルの時間発展図
    calc_S_kpq.ipynb          # 対称化非線形エネルギー伝達関数の評価
    plot_S_kpq_graph.ipynb    # 対称化非線形エネルギー伝達関数のネットワーク可視化
    ```
