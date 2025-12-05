import pytest
import torch
import sys
import os

# srcフォルダのパス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

@pytest.fixture
def model_config():
    """モデルの基本設定
    テストを高速化するため、層の深さや埋め込み次元を小さくしています。
    """
    return {
        'img_size': 224,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 3,
        'embed_dim': 96,
        'depths': [2, 2, 2],         
        'depths_decoder': [1, 2, 2],
        'num_heads': [3, 6, 12],
        'window_size': 7
    }

@pytest.fixture
def model(model_config):
    """モデルのインスタンス化"""
    return SwinTransformerSys(**model_config)

def test_forward_output_shape(model, model_config):
    """
    [テスト1] 正常なフォワードパスと出力サイズの確認
    Input: (B, 3, 224, 224) -> Output: (B, num_classes, 224, 224)
    """
    B = 2
    H = model_config['img_size']
    W = model_config['img_size']
    inputs = torch.randn(B, model_config['in_chans'], H, W)
    
    outputs = model(inputs)
    
    # セグメンテーションなので入力と同じH, Wに戻るはず
    expected_shape = (B, model_config['num_classes'], H, W)
    
    assert outputs.shape == expected_shape, \
        f"Output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

def test_backward_pass(model, model_config):
    """
    [テスト2] バックプロパゲーション(学習)が可能かの確認
    勾配が途切れていないかをチェックします。
    """
    B = 2
    H = model_config['img_size']
    W = model_config['img_size']
    inputs = torch.randn(B, model_config['in_chans'], H, W)
    # ターゲット(正解ラベル)を作成
    targets = torch.randint(0, model_config['num_classes'], (B, H, W))
    
    outputs = model(inputs)
    
    # 損失関数の計算
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    
    # 勾配計算
    loss.backward()
    
    # 最初のパラメータに勾配が乗っているかチェック
    first_layer_param = next(model.parameters())
    assert first_layer_param.grad is not None, "Gradients not computed. Check detach() or requires_grad."

def test_dimension_mismatch(model_config):
    """
    [テスト3] 不正な入力サイズに対する挙動確認
    Swin Transformerは (window_size * patch_size) の倍数でないと
    PatchEmbedなどでエラーになる設計になっているか確認。
    """
    # インスタンス化
    model = SwinTransformerSys(**model_config)
    
    # 225は 4(patch)の倍数ではないのでエラーになるはず
    invalid_size = 225
    inputs = torch.randn(1, 3, invalid_size, invalid_size)
    
    # ここでは実行時エラー(RuntimeError)またはAssertionErrorを期待
    with pytest.raises((RuntimeError, AssertionError)):
        model(inputs)