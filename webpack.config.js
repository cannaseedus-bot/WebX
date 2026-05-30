const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

const isProduction = process.env.NODE_ENV === 'production';

module.exports = {
    mode: isProduction ? 'production' : 'development',
    entry: './src/index.js',
    output: {
        filename: isProduction ? '[name].[contenthash].js' : '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
        clean: true,
    },
    devtool: isProduction ? 'source-map' : 'inline-source-map',
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/, 
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env'],
                    },
                },
            },
        ],
    },
    plugins: [
        new CleanWebpackPlugin(),
        // src/index.html is the self-contained WebX-3D app (inline scripts).
        // inject:false — don't auto-inject the library bundle into the app HTML.
        new HtmlWebpackPlugin({
            title: 'KUHUL WebX-3D',
            template: 'src/index.html',
            inject: false,
            filename: 'index.html',
        }),
        // Also emit the library bundle as webx3d.bundle.js for external consumers.
    ],
    devServer: {
        static: './dist',
        hot: true,
    },
};