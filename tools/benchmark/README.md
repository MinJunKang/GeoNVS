# :bar_chart: Benchmark

We provide <a href="https://github.com/Stability-AI/stable-virtual-camera/releases/tag/benchmark">in this release</a> (`benchmark.zip`) with the following 17 entries as a benchmark to evaluate NVS models.
We hope this will help standardize the evaluation of NVS models and facilitate fair comparison between different methods.

<table>
  <thead>
    <tr>
      <th align="center">Dataset</th>
      <th align="center">Split</th>
      <th align="center">Path</th>
      <th align="center">Content</th>
      <th align="center">Image Preprocessing</th>
      <th align="center">Image Postprocessing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">OmniObject3D</td>
      <td align="center"><code>S</code> (SV3D), <code>O</code> (Ours) </td>
      <td align="center"><code>omniobject3d</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center">GSO</td>
      <td align="center"><code>S</code> (SV3D), <code>O</code> (Ours) </td>
      <td align="center"><code>gso</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center" rowspan="4">RealEstate10K</td>
      <td align="center"><code>D</code> (4DiM) </td>
      <td align="center"><code>re10k-4dim</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">resize to 256</td>
    </tr>
    <tr>
      <td align="center"><code>R</code> (ReconFusion) </td>
      <td align="center"><code>re10k</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center"><code>P</code> (pixelSplat) </td>
      <td align="center"><code>re10k-pixelsplat</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">resize to 256</td>
    </tr>
    <tr>
      <td align="center"><code>V</code> (ViewCrafter) </td>
      <td align="center"><code>re10k-viewcrafter</code></td>
      <td align="center"><code>images/*.png</code>,<code>transforms.json</code>,<code>train_test_split_*.json</code></td>
      <td align="center">resize the shortest side to 576 (<code>--L_short 576</code>)</td>
      <td align="center">center crop</td>
    </tr>
    <tr>
      <td align="center">LLFF</td>
      <td align="center"><code>R</code> (ReconFusion) </td>
      <td align="center"><code>llff</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center">DTU</td>
      <td align="center"><code>R</code> (ReconFusion) </td>
      <td align="center"><code>dtu</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">CO3D</td>
      <td align="center"><code>R</code> (ReconFusion) </td>
      <td align="center"><code>co3d</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center"><code>V</code> (ViewCrafter) </td>
      <td align="center"><code>co3d-viewcrafter</code></td>
      <td align="center"><code>images/*.png</code>,<code>transforms.json</code>,<code>train_test_split_*.json</code></td>
      <td align="center">resize the shortest side to 576 (<code>--L_short 576</code>)</td>
      <td align="center">center crop</td>
    </tr>
    <tr>
      <td align="center" rowspan="2" >WildRGB-D</td>
      <td align="center"><code>Oₑ</code> (Ours, easy) </td>
      <td align="center"><code>wildgbd/easy</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center"><code>Oₕ</code> (Ours, hard) </td>
      <td align="center"><code>wildgbd/hard</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center">Mip-NeRF360</td>
      <td align="center"><code>R</code> (ReconFusion) </td>
      <td align="center"><code>mipnerf360</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">DL3DV-140</td>
      <td align="center"><code>O</code> (Ours) </td>
      <td align="center"><code>dl3dv10</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center"><code>L</code> (Long-LRM) </td>
      <td align="center"><code>dl3dv140</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Tanks and Temples</td>
      <td align="center"><code>V</code> (ViewCrafter) </td>
      <td align="center"><code>tnt-viewcrafter</code></td>
      <td align="center"><code>images/*.png</code>,<code>transforms.json</code>,<code>train_test_split_*.json</code></td>
      <td align="center">resize the shortest side to 576 (<code>--L_short 576</code>)</td>
      <td align="center">center crop</td>
    </tr>
    <tr>
      <td align="center"><code>L</code> (Long-LRM) </td>
      <td align="center"><code>tnt-longlrm</code></td>
      <td align="center"><code>train_test_split_*.json</code></td>
      <td align="center">center crop to 576</td>
      <td align="center">\</td>
    </tr>
  </tbody>
</table>

- For entries without `images/*.png` and `transforms.json`, we use the images from the original dataset after converting them into the `reconfusion` format, which is then parsable by `ReconfusionParser` (`seva/data_io.py`).
  Please note that during this conversion, you should sort the images by `sorted(image_paths)`, which is then directly indexable by our train/test ids. We provide in `benchmark/export_reconfusion_example.py` an example script converting an existing academic dataset into the the scene folders.
- For evaluation and benchmarking, we first conduct operations in the `Image Preprocessing` column to the model input and then operations in the `Image Postprocessing` column to the model output. The final processed samples are used for metric computation.

## Acknowledgment

We would like to thank Wangbo Yu, Aleksander Hołyński, Saurabh Saxena, and Ziwen Chen for their kind clarification on experiment settings.
