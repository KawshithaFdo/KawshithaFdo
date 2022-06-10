function testModel(model,inputData,normalizationData){
    const { inputmax, inputmin, labelmin, labelmax } =
    normalizationData;

    const[xs, preds] = tf.tidy(() =>{
        const xs = tf.linspace(0,1,100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs
        .mul(inputmax.sub(inputmin))
        .add(inputmin);

        const unNormPreds=preds
        .mul(labelmax.sub(labelmin))
        .add(labelmin);

        return[unNormXs.dataSync(),unNormPreds.dataSync()];
    });

    const predictedPoints=Array.from(xs).map((val,i)=>{
        return{ x: val, y:preds[i] }
    });

    const originalPoints=inputData.map(d=>({
        x: d.hp, y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name:'Model Predictions vs Original Data'},
        {values : [originalPoints,predictedPoints],series :
        ['original','predicted'] },
        {
            xLabel:'Horsepower',
            yLabel:'MPG',
            height:300
        }
    );


}

async function trainModel(model,inputs,labels){
    model.compile(
        {
            optimizer:tf.train.adam(),
            loss:tf.losses.meanSquaredError,
            metrics:['mse']
        }
    );

    const batchSize=32;
    const epochs=50;

    return await model.fit(inputs,labels,{
        batchSize,
        epochs,
        shuffle:true,
        callbacks:tfvis.show.fitCallbacks(
            {name:"Training performance monitor"},
            ["loss","mse"],
            {height:200,callbacks:['onEpochEnd']}
            )
    })
}
function dataToTensor(data){
    return tf.tidy(()=>{
        tf.util.shuffle(data)

        const inputs= data.map(d=>d.hp);
        const labels= data.map(d=>d.mpg);

        const intputTensor=tf.tensor2d(inputs,[inputs.length,1])
        const labelTensor=tf.tensor2d(labels,[labels.length,1])

        const inputmax=intputTensor.max();
        const inputmin=intputTensor.min();
        const labelmax=labelTensor.max();
        const labelmin=labelTensor.min();

        const normalizeInputs = intputTensor.sub(inputmin).div(inputmax.sub(inputmin))
        const normalizelabels = labelTensor.sub(labelmin).div(labelmax.sub(labelmin))

        return{
            inputs:normalizeInputs,
            labels:normalizelabels,
            inputmax,
            inputmin,
            labelmax,
            labelmin
        }
    });
}
async function createModel(){
    const model=tf.sequential(); 
    model.add(tf.layers.dense(
        {
            inputShape:[1],
            units:1
        }
    ));
    model.add(tf.layers.dense(
        {
            units:1
        }
    ));
    return model;
}
async function getData(){
    const dataResp = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json")
    const rawData=await dataResp.json();

    const data=rawData.map(function(cardata){
        return {
            hp:cardata.Horsepower,
            mpg:cardata.Miles_per_Gallon
        }
    });
    return data;
} 

async function run(){
    const data=await getData();
    const values=data.map(d=>{
        return{
            x:d.hp,
            y:d.mpg
        }
    })
    tfvis.render.scatterplot({name:'Horsepower Vs Miles per gallon'},
    {values},
    {
        xLabel:"Horsepower",
        yLabel:"Mile per gallon",
        height:450
    });

    const model=await createModel();
    tfvis.show.modelSummary({name:'HPvsMPG model summery'},model);


    const tensorData=dataToTensor(data);
    const {inputs,labels} = tensorData;
    await trainModel(model,inputs,labels);
    //console.log("Done");
    //testModel(model,data,tensorData);

    const newDataPoint=tf.tensor2d([50],[1,1]);
    const newDataPointNorm=newDataPoint.sub(inputmin).div(inputmax.sub(inputmin))


    const predValue=model.predict(newDataPointNorm);
    const xPred= await newDataPointNorm.data();

    const  yPredUnNormalize=predValue.mul(inputmax.sub(inputmin)).add(inputmin)
    const yPred=await yPredUnNormalize.data();
    console.log(xPred,yPred);
}
document.addEventListener("DOMContentLoaded", run);