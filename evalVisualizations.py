# ====== CUQ USER SURVEY VISUALISATIONS ======
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generateHeatmap():
    """Generate heatmap visualization of survey responses"""
        
    # CUQ data
    cuqResponses = {
        'Question 1': [5, 4, 4, 4, 4, 4, 4],
        'Question 2': [3, 2, 1, 1, 2, 2, 2],
        'Question 3': [4, 5, 5, 4, 3, 5, 4],
        'Question 4': [1, 1, 1, 2, 4, 1, 2],
        'Question 5': [4, 5, 5, 5, 1, 4, 3],
        'Question 6': [1, 1, 1, 2, 4, 2, 3],
        'Question 7': [4, 4, 4, 4, 2, 4, 2],
        'Question 8': [3, 2, 3, 3, 5, 3, 1],
        'Question 9': [5, 4, 4, 3, 2, 4, 4],
        'Question 10': [2, 2, 2, 2, 4, 2, 2],
        'Question 11': [4, 5, 5, 4, 4, 5, 5],
        'Question 12': [1, 1, 2, 2, 1, 1, 1],
        'Question 13': [4, 3, 4, 4, 4, 3, 4],
        'Question 14': [1, 1, 2, 3, 4, 3, 1],
        'Question 15': [4, 5, 5, 4, 4, 4, 4],
        'Question 16': [2, 1, 3, 2, 1, 1, 3]
    }
    
    # Count responses for each value (1-5) per question
    data = {
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': []
    }
    
    for question, responses in cuqResponses.items():
        for value in [1, 2, 3, 4, 5]:
            count = responses.count(value)
            data[str(value)].append(count)
    
    # Create dataframe with questions as rows, response values as columns
    df = pd.DataFrame(data, index=list(cuqResponses.keys()))
    
    # Create the heatmap
    plt.figure(figsize=(8, 10))
    sns.heatmap(df, annot=True, fmt='g', cmap='YlGnBu', cbar_kws={'label': 'Count'})
    
    # Add title and labels
    plt.title('CUQ Response Distribution Heatmap')
    plt.xlabel('Response Value')
    plt.ylabel('Questions')
    plt.tight_layout()
    
    # Save the plot
    outputFile = 'survey_heatmap.png'
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as '{outputFile}'")
    plt.close()

def generateStackedBarChart():
    """Generate stacked bar chart visualization of Likert scale responses"""
        
    # CUQ responses
    cuqResponses = {
        'Question 1': [5, 4, 4, 4, 4, 4, 4],
        'Question 2': [3, 2, 1, 1, 2, 2, 2],
        'Question 3': [4, 5, 5, 4, 3, 5, 4],
        'Question 4': [1, 1, 1, 2, 4, 1, 2],
        'Question 5': [4, 5, 5, 5, 1, 4, 3],
        'Question 6': [1, 1, 1, 2, 4, 2, 3],
        'Question 7': [4, 4, 4, 4, 2, 4, 2],
        'Question 8': [3, 2, 3, 3, 5, 3, 1],
        'Question 9': [5, 4, 4, 3, 2, 4, 4],
        'Question 10': [2, 2, 2, 2, 4, 2, 2],
        'Question 11': [4, 5, 5, 4, 4, 5, 5],
        'Question 12': [1, 1, 2, 2, 1, 1, 1],
        'Question 13': [4, 3, 4, 4, 4, 3, 4],
        'Question 14': [1, 1, 2, 3, 4, 3, 1],
        'Question 15': [4, 5, 5, 4, 4, 4, 4],
        'Question 16': [2, 1, 3, 2, 1, 1, 3]
    }
        
    # Count responses for each Likert category
    data = {
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': []
    }
        
    for question, responses in cuqResponses.items():
        for score in [1, 2, 3, 4, 5]:
            data[str(score)].append(responses.count(score))
        
    df = pd.DataFrame(data, index=list(cuqResponses.keys()))
    
    # Create the stacked bar chart
    ax = df.plot(kind='bar', stacked=True, figsize=(14, 6),
                 color=['#d62728', '#ff7f0e', '#ffdd57', '#98df8a', '#2ca02c'])
    
    # Add title and labels
    plt.title('CUQ Responses - Stacked Bar Chart')
    plt.xlabel('Questions')
    plt.ylabel('Number of Responses')
    plt.legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    outputFile = 'survey_stacked_bar.png'
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    print(f"Stacked bar chart saved as '{outputFile}'")
    plt.close()

def evaluateVisualizations():
    print("Usability Testing Visualizations")
    print()
    
    print("Generating heatmap...")
    generateHeatmap()
    print()
    
    print("Generating stacked bar chart...")
    generateStackedBarChart()
    print()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    evaluateVisualizations()
