Data Visualization

sns.heatmap(MTW_bulk,xticklabels = list(bulk[:,0]),yticklabels = list(bulk[:,0]),linewidths = 0.1,linecolor = 'black',cmap="YlGnBu")
plt.yticks(rotation = 0,fontsize = 5)
plt.xticks(rotation = 90,fontsize = 5)
plt.title('Distance matrix of MYC genes from Bulk data')
plt.savefig('MYC_genes_61_61',dpi = 400)
plt.show()
