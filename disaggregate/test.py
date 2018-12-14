

        house_data = pd.read_csv(os.path.join(REDD_DIR, 'building_{0}.csv'.format(self.house_id)))


        truth_df['Main'] = np.sum(truth_df.values, axis=1)
