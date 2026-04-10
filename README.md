# Marketing Campaign Performance Analysis

A data-driven evaluation of 200,000 marketing campaigns across six channels, five companies, and five U.S. markets. This project analyzes multi-channel marketing effectiveness using statistical testing, machine learning, and visual analytics.

## Key Highlights

- **200,000 campaigns** analyzed across Email, Google Ads, YouTube, Instagram, Facebook, and Website channels
- **ANOVA hypothesis testing** to validate which factors significantly impact ROI
- **Random Forest predictive model** (200 trees) to identify the most influential campaign features
- **14 publication-quality visualizations** embedded in a professional report
- **17-section Word document** with executive summary, methodology, findings, and strategic recommendations

## Tools & Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Analysis** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **Statistical Testing** | scipy (one-way ANOVA) |
| **Machine Learning** | scikit-learn (Random Forest Regressor) |
| **Report Generation** | python-docx |

## Project Structure

```
marketing-campaign-analysis/
├── README.md
├── campaign_analysis/
│   ├── analyze.py                                    # Full analysis pipeline
│   ├── Marketing_Campaign_Performance_Analysis.docx  # Final report
│   └── figures/                                      # All generated charts
│       ├── 01_channel_overview.png
│       ├── 02_campaign_type_roi.png
│       ├── 03_channel_campaign_heatmap.png
│       ├── 04_audience_analysis.png
│       ├── 05_cost_efficiency.png
│       ├── 06_monthly_trends.png
│       ├── 07_customer_segments.png
│       ├── 08_company_comparison.png
│       ├── 09_duration_impact.png
│       ├── 10_location_performance.png
│       ├── 11_social_media_deep_dive.png
│       ├── 12_feature_importance.png
│       ├── 13_predicted_vs_actual.png
│       └── 14_correlation_matrix.png
```

## Dataset

**Source:** [Marketing Campaign Performance Dataset](https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset) (Kaggle, CC0 License)

To reproduce the analysis:
1. Download the dataset from Kaggle and place `marketing_campaign_dataset.csv` in the project root
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy scikit-learn python-docx`
3. Run: `python campaign_analysis/analyze.py`

## Report Sections

1. Executive Summary
2. Introduction & Objectives
3. Data Overview & Methodology
4. Channel Performance Analysis
5. Campaign Type Effectiveness
6. Social Media Deep-Dive (Instagram vs. Facebook vs. YouTube)
7. Audience & Customer Segment Analysis
8. Geographic & Demographic Insights
9. Cost Efficiency & Budget Optimization
10. Temporal Trends & Seasonality
11. Company Benchmarking
12. Statistical Significance Testing (ANOVA)
13. Predictive Modeling & Feature Importance
14. Correlation Analysis
15. Key Findings & Strategic Recommendations
16. Limitations & Future Research
17. Appendix

## Key Findings

- All six channels deliver comparable ROI (~5.0x), indicating a well-diversified marketing portfolio
- ANOVA testing identifies which grouping variables show statistically significant ROI differences
- The Random Forest model reveals that observable campaign features explain very little ROI variance, suggesting unmeasured factors (creative quality, ad copy) are the true performance drivers
- Campaign duration has negligible impact on ROI, favoring shorter, more iterative campaign cycles

## Author

Rebecca Wu
