import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class UnifiedGenomicRegionAnalyzer:
    def __init__(self, gtf_df: pd.DataFrame, region_types: List[str] = None):
        self.gtf_df = gtf_df
        default = ['CDS', '5UTR', '3UTR', 'intron', 'stop_codon', 'start_codon']
        self.region_types = region_types if region_types is not None else default
        self.gene_regions = None
        # Auto infer introns
        self._infer_intron()
        # Auto preprocess gtf
        self._preprocess_gtf()
    
    def _preprocess_gtf(self):
        gene_regions_temp = {}
        for _, row in tqdm(self.gtf_df.iterrows(), total=len(self.gtf_df), desc="Processing GTF"):
            gene_id = row['gene_id']
            if pd.isna(gene_id):
                continue
            
            chrom = row['chrom']
            start = int(row['start'])
            end = int(row['end'])
            strand = row['strand']
            feature = row['feature']
            
            if feature in ['five_prime_utr', 'five_prime_UTR', '5_prime_utr']:
                feature = '5UTR'
            elif feature in ['three_prime_utr', 'three_prime_UTR', '3_prime_utr']:
                feature = '3UTR'
            elif feature not in self.region_types:
                continue
            
            if chrom not in gene_regions_temp:
                gene_regions_temp[chrom] = {}
            
            region_key = (start, end, strand, gene_id)
            if region_key not in gene_regions_temp[chrom]:
                gene_regions_temp[chrom][region_key] = []
                
            gene_regions_temp[chrom][region_key].append(feature)
        
        self.gene_regions = gene_regions_temp
        print(f"Finished, containing {len(self.gene_regions)} chromosomes with gene regions.")
        try:
            pickle.dumps(self.gene_regions)
            print("Pass pickle test for GTF data structure.")
        except Exception as e:
            print(f"Failed pickle test {e}")
            
    def _infer_intron(self):
        extend_gtf_df = self.gtf_df.copy()
        new_rows = []
        for chrom in self.gtf_df['chrom'].unique():
            chrom_data = self.gtf_df[self.gtf_df['chrom'] == chrom]
            intron_rows = self._infer_intron_for_chrom(chrom_data)
            new_rows.extend(intron_rows)
        if new_rows:
            new_gtf = pd.DataFrame(new_rows)
            for col in self.gtf_df.columns:
                if col not in new_gtf.columns:
                    new_gtf[col] = np.nan
            extended_gtf = pd.concat([self.gtf_df, new_gtf], ignore_index=True)
            self.gtf_df = extended_gtf
        
    def _infer_intron_for_chrom(self, chrom_data: pd.DataFrame):
        intron_rows = []
        for gene_id in chrom_data['gene_id'].unique():
            gene_data = chrom_data[chrom_data['gene_id'] == gene_id]
            for transcript_id in gene_data.get('transcript_id', [gene_id]).unique():
                if pd.isna(transcript_id):
                    transcript_id = gene_id
                    
                transcript_data = gene_data
                if 'transcript_id' in gene_data.columns:
                    transcript_data = gene_data[gene_data['transcript_id'] == transcript_id]
                exon_data = transcript_data[transcript_data['feature'] == 'exon']
                if len(exon_data) > 1: 
                    exons = []
                    strand = exon_data['strand'].iloc[0]
                    for _, row in exon_data.iterrows():
                        exons.append((int(row['start']), int(row['end'])))
                    exons.sort()
                    
                    for i in range(len(exons) - 1):
                        intron_start = exons[i][1] + 1
                        intron_end = exons[i+1][0] - 1
                        if intron_start < intron_end:
                            intron_row = {
                                'chrom': exon_data['chrom'].iloc[0],
                                'start': intron_start,
                                'end': intron_end,
                                'strand': strand,
                                'feature': 'intron',
                                'gene_id': gene_id,
                                'transcript_id': transcript_id 
                            }
                            intron_rows.append(intron_row)
        return intron_rows

    def _create_region_masks_base_level(self, chrom, start, end, strand, seq_len):
        """
        Base-level分析：每个位置对应一个碱基
        """
        length = end - start
        
        # 创建碱基位置数组
        if seq_len == length:
            # 1:1对应
            base_positions = np.arange(start, end)
        else:
            # 需要插值映射
            base_positions = np.linspace(start, end - 1, seq_len).astype(int)
        
        if strand == '-':
            base_positions = base_positions[::-1]
            
        region_masks = {region: np.zeros(seq_len, dtype=bool) for region in self.region_types}
        
        if self.gene_regions is None or chrom not in self.gene_regions:
            return region_masks
        
        # 为每个碱基位置标注区域类型
        for (g_start, g_end, g_strand, gene_id), features in self.gene_regions[chrom].items():
            for i, pos in enumerate(base_positions):
                if g_start <= pos <= g_end:
                    for feature in features:
                        if feature in self.region_types:
                            region_masks[feature][i] = True
        
        return region_masks

    def _create_region_masks_bin_level(self, chrom, start, end, strand, seq_len):
        """
        Bin-level分析：每个bin覆盖一定的基因组范围
        """
        length = end - start
        bin_size = length / seq_len
        
        # 计算每个bin的中心位置
        bin_centers = np.linspace(start + bin_size/2, end - bin_size/2, seq_len)
        
        if strand == '-':
            bin_centers = bin_centers[::-1]
            
        region_masks = {region: np.zeros(seq_len, dtype=bool) for region in self.region_types}
        
        if self.gene_regions is None or chrom not in self.gene_regions:
            return region_masks
        
        # 为每个bin标注区域类型
        for (g_start, g_end, g_strand, gene_id), features in self.gene_regions[chrom].items():
            for i, bin_center in enumerate(bin_centers):
                bin_start = bin_center - bin_size/2
                bin_end = bin_center + bin_size/2
                
                # 检查bin是否与基因区域有重叠
                if not (bin_end < g_start or bin_start > g_end):
                    for feature in features:
                        if feature in self.region_types:
                            # 对于start_codon和stop_codon，只要有重叠就算
                            if feature in ['start_codon', 'stop_codon']:
                                region_masks[feature][i] = True
                            # 对于其他区域，需要bin中心在区域内
                            elif g_start <= bin_center <= g_end:
                                region_masks[feature][i] = True
        
        return region_masks
    
    def _extract_data_unified(self, signal_array, chrom, start, end, strand, label, 
                             gene_id=None, analysis_level='auto'):
        """
        统一的数据提取方法
        
        Parameters:
        analysis_level: str, 'base', 'bin', or 'auto'
            - 'base': 碱基级别分析
            - 'bin': bin级别分析  
            - 'auto': 根据序列长度自动判断
        """
        if isinstance(signal_array, torch.Tensor):
            signal_array = signal_array.squeeze().cpu().numpy()
        elif isinstance(signal_array, np.ndarray):
            signal_array = signal_array.squeeze()
        
        original_signal = signal_array.copy()
        seq_len = len(signal_array)
        genomic_length = end - start
        
        # 自动判断分析级别
        if analysis_level == 'auto':
            if seq_len == genomic_length:
                analysis_level = 'base'
                print(f"Auto-detected: base-level analysis (seq_len={seq_len}, genomic_len={genomic_length})")
            else:
                analysis_level = 'bin'
                print(f"Auto-detected: bin-level analysis (seq_len={seq_len}, genomic_len={genomic_length})")
        
        if strand == '-':
            signal_array = signal_array[::-1]
        
        # 根据分析级别创建mask
        if analysis_level == 'base':
            region_masks = self._create_region_masks_base_level(chrom, start, end, strand, seq_len)
        elif analysis_level == 'bin':
            region_masks = self._create_region_masks_bin_level(chrom, start, end, strand, seq_len)
        else:
            raise ValueError(f"Unknown analysis_level: {analysis_level}")
        
        unified_data = {
            'label': label,
            'gene_id': gene_id,
            'chrom': chrom,
            'start': start,
            'end': end,
            'strand': strand,
            'signal': signal_array,
            'original_signal': original_signal,
            'region_masks': region_masks,
            'seq_len': seq_len,
            'genomic_length': genomic_length,
            'analysis_level': analysis_level
        }
        
        return unified_data
    
    def _calculate_gene_enrichment_unified(self, unified_data):
        """
        从统一数据结构计算基因内部富集分数
        """
        signal = unified_data['signal']
        region_masks = unified_data['region_masks']
        
        gene_mean = np.mean(signal)
        gene_std = np.std(signal)
        
        region_enrichments = {}
        for region_type in self.region_types:
            mask = region_masks[region_type]
            if np.any(mask):
                region_signal = signal[mask]
                region_mean = np.mean(region_signal)
                
                if gene_std > 1e-8:
                    z_score = (region_mean - gene_mean) / gene_std
                else:
                    z_score = 0.0
                
                region_enrichments[region_type] = z_score
            else:
                region_enrichments[region_type] = 0.0
        
        gene_enrichment = {
            'label': unified_data['label'],
            'gene_id': unified_data['gene_id'],
            'chrom': unified_data['chrom'],
            'start': unified_data['start'],
            'end': unified_data['end'],
            'strand': unified_data['strand'],
            'analysis_level': unified_data['analysis_level'],
            **region_enrichments
        }
        
        return gene_enrichment
    
    def _extract_bin_collections_from_unified(self, unified_data_list):
        """
        从统一数据结构提取bin collections（向后兼容）
        """
        bin_collections = {}
        
        for data in unified_data_list:
            label = data['label']
            signal = data['signal']
            region_masks = data['region_masks']
            
            if label not in bin_collections:
                bin_collections[label] = {region: [] for region in self.region_types}
            
            for region_type in self.region_types:
                mask = region_masks[region_type]
                if np.any(mask):
                    region_signal = signal[mask]
                    bin_collections[label][region_type].extend(region_signal.tolist())
        
        return bin_collections
    
    def _process_single_data_unified(self, data, analysis_level='auto'):
        """统一的单数据处理方法"""
        try:
            unified_data = self._extract_data_unified(
                data['signal_array'], data['chrom'], data['start'], 
                data['end'], data['strand'], data['label'], 
                data.get('gene_id', None), analysis_level
            )
            
            gene_enrichment = self._calculate_gene_enrichment_unified(unified_data)
            
            return {
                'unified_data': unified_data,
                'gene_enrichment': gene_enrichment
            }
        except Exception as e:
            print(f"Error processing data: {e}")
            return {
                'unified_data': None,
                'gene_enrichment': None
            }
    
    def batch_process_signals_unified(self, signal_data_list, analysis_level='auto', n_jobs=1):
        """
        统一的批处理方法
        
        Parameters:
        analysis_level: str, 'base', 'bin', or 'auto'
        
        Returns:
        unified_data_list: list, 统一格式的数据
        gene_enrichment_df: pd.DataFrame, 基因级别富集
        bin_collections: dict, bin collections（向后兼容）
        """
        print(f"Processing {len(signal_data_list)} samples for {analysis_level}-level analysis using {n_jobs} threads.")
        
        results = []
        
        if n_jobs == 1 or len(signal_data_list) < 10:
            for data in tqdm(signal_data_list, desc="Processing signals"):
                result = self._process_single_data_unified(data, analysis_level)
                results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(self._process_single_data_unified, data, analysis_level) 
                          for data in signal_data_list]
                for future in tqdm(futures, desc="Processing signals"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error in future result: {e}")
                        results.append({'unified_data': None, 'gene_enrichment': None})
        
        # 分离结果
        unified_data_list = []
        gene_enrichment_data = []
        
        for result in results:
            if result['unified_data'] is not None:
                unified_data_list.append(result['unified_data'])
            if result['gene_enrichment'] is not None:
                gene_enrichment_data.append(result['gene_enrichment'])
        
        # 创建基因富集DataFrame
        if gene_enrichment_data:
            gene_enrichment_df = pd.DataFrame(gene_enrichment_data)
        else:
            gene_enrichment_df = pd.DataFrame()
        
        # 创建bin collections（向后兼容）
        bin_collections = self._extract_bin_collections_from_unified(unified_data_list)
        
        return unified_data_list, gene_enrichment_df, bin_collections
    
    def aggregate_gene_enrichment_by_label(self, gene_enrichment_df):
        """按label聚合基因级别的富集数据"""
        if gene_enrichment_df.empty:
            return pd.DataFrame()
            
        numeric_cols = [col for col in self.region_types if col in gene_enrichment_df.columns]
        
        if not numeric_cols:
            print("Warning: No region type columns found in gene_enrichment_df")
            return pd.DataFrame()
        
        # 聚合并添加统计信息
        aggregated_df = gene_enrichment_df.groupby('label')[numeric_cols].agg('mean').round(4)
        gene_counts = gene_enrichment_df.groupby('label').size()
        aggregated_df['n_genes'] = gene_counts
        
        # 添加分析级别信息
        if 'analysis_level' in gene_enrichment_df.columns:
            analysis_levels = gene_enrichment_df.groupby('label')['analysis_level'].first()
            aggregated_df['analysis_level'] = analysis_levels
        
        return aggregated_df
    
    def get_unified_statistics(self, unified_data_list):
        """获取统一格式数据的统计信息"""
        stats_data = []
        
        for data in unified_data_list:
            label = data['label']
            gene_id = data.get('gene_id', 'unknown')
            region_masks = data['region_masks']
            signal = data['signal']
            analysis_level = data['analysis_level']
            
            row = {
                'label': label,
                'gene_id': gene_id,
                'analysis_level': analysis_level,
                'sequence_length': len(signal),
                'genomic_length': data['genomic_length'],
                'total_signal': np.sum(signal),
                'mean_signal': np.mean(signal)
            }
            
            # 每个区域的统计
            for region_type in self.region_types:
                mask = region_masks[region_type]
                if np.any(mask):
                    region_signal = signal[mask]
                    row[f'{region_type}_positions'] = np.sum(mask)
                    row[f'{region_type}_signal_sum'] = np.sum(region_signal)
                    row[f'{region_type}_signal_mean'] = np.mean(region_signal)
                else:
                    row[f'{region_type}_positions'] = 0
                    row[f'{region_type}_signal_sum'] = 0
                    row[f'{region_type}_signal_mean'] = 0
            
            stats_data.append(row)
        
        return pd.DataFrame(stats_data)
    
    def create_enrichment_heatmap(self, enrichment_df, title="Genomic Region Enrichment", 
                                figsize=(10, 8), cmap='RdBu_r', center=0):
        """创建富集热图"""
        available_regions = [col for col in self.region_types if col in enrichment_df.columns]
        if not available_regions:
            print("Error: No region columns found in enrichment_df")
            return None, None
            
        plot_data = enrichment_df[available_regions]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(plot_data, annot=True, cmap=cmap, center=center, 
                   fmt='.2f', cbar_kws={'label': 'Z-score Enrichment'}, ax=ax)
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Genomic Regions', fontsize=12)
        ax.set_ylabel('Sample Labels', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig, ax
    
    # ============ 向后兼容的方法 ============
    
    def batch_process_signals_dual(self, signal_data_list, n_jobs=1):
        """向后兼容方法：双层次分析"""
        unified_data_list, gene_enrichment_df, _ = self.batch_process_signals_unified(
            signal_data_list, analysis_level='auto', n_jobs=n_jobs
        )
        
        # 分离base_level_data格式
        base_level_data = []
        for data in unified_data_list:
            base_data = {
                'label': data['label'],
                'gene_id': data['gene_id'],
                'chrom': data['chrom'],
                'start': data['start'],
                'end': data['end'],
                'strand': data['strand'],
                'signal': data['signal'],
                'original_signal': data['original_signal'],
                'region_masks': data['region_masks'],
                'seq_len': data['seq_len']
            }
            base_level_data.append(base_data)
        
        return base_level_data, gene_enrichment_df
    
    def batch_process_signals_to_bins(self, signal_data_list, n_jobs=1):
        """向后兼容方法：bin级别分析"""
        _, _, bin_collections = self.batch_process_signals_unified(
            signal_data_list, analysis_level='bin', n_jobs=n_jobs
        )
        return bin_collections
    
    def get_baselevel_stats(self, base_level_data):
        """向后兼容方法：base级别统计"""
        return self.get_unified_statistics(base_level_data)