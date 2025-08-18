const fs = require("fs");
const csv = require("csv-parser");
const axios = require("axios");
const OSS = require("ali-oss");

// 配置阿里云OSS
const client = new OSS({
  region: "",
  accessKeyId: "",
  accessKeySecret: "",
  bucket: "",
});

// 目标CSV（从平台导出）
const targetFile = "2025-08-17T13_52_55.957510Z-report.csv";

const successBaseUrl = "ots/202508_tengyun_android/success_test";
const retryBaseUrl = "ots/202508_tengyun_android/retry_success_test";

async function deleteDir(dir) {
  const result = await client.listV2({
    prefix: `${dir}/`,
    delimiter: "/",
  });

  if (result.objects.length > 0) {
    console.log("delete:", dir, result.objects.length);
    await client.deleteMulti(result.objects.map((obj) => obj.name));
  }
}

function actionTypeMap(action) {
  let type = "unknown";
  switch (action.type) {
    case "click":
      type = "click";
      if (action.click_type === "long press") {
        type = "long_press";
      }
      if (action.click_times === "double click") {
        type = "double_tap";
      }
      break;
    case "swipe":
      type = "scroll";
      break;
    case "input":
      type = "input_text";
      break;
    case "end":
      type = "status";
      break;
    case "open_app":
      type = "open_app";
      break;
    case "wait":
      type = "wait";
      break;

    default:
      break;
  }
  return { type };
}

function getDirection(action) {
  if (action.type !== "swipe") {
    return null;
  }
  const { start_position, end_position } = action;
  if (!start_position || !end_position) {
    return null;
  }

  const dx = end_position[0] - start_position[0];
  const dy = end_position[1] - start_position[1];

  const absDx = Math.abs(dx);
  const absDy = Math.abs(dy);

  if (absDy > absDx) {
    return dy > 0 ? "down" : "up";
  } else {
    return dx > 0 ? "right" : "left";
  }
}

function revertPosition(position) {
  if (!position) {
    return null;
  }
  try {
    return JSON.parse(position);
  } catch (error) {
    return null;
  }
}

function getPosition(action, isStartPoint) {
  let position = null;
  switch (action.type) {
    case "click":
      position = action.position;
      break;
    case "swipe":
      position = isStartPoint ? action.start_position : action.end_position;
      break;

    default:
      break;
  }
  return position;
}

async function processRow(row) {
  try {
    const { id, json: jsonUrl, result, instruction } = row;
    // if (id !== "4857") {
    //   return;
    // }
    const parsedResult = JSON.parse(result);
    const { trajectory_type, trajectory_think } = parsedResult;

    // 这里处理一下10以内的url，把 01/task.json 改成 1/task.json
    const regex = /\/0(\d)\/task\.json/;
    const dataUrl = jsonUrl.replace(regex, (match, p1) => `/${parseInt(p1)}/task.json`);

    // 当前行文件目录基准
    const url = new URL(dataUrl);
    const recordPath = url.pathname.split("/").slice(1, -1).join("/");
    // 要复制的oss目标目录
    const targetBasePath = trajectory_type === 0 ? successBaseUrl : retryBaseUrl;

    // 下载JSON文件
    const response = await axios.get(dataUrl, { responseType: "json" });
    const jsonData = response.data;

    // 目标目录
    const targetDir = `${targetBasePath}/${id}`;

    // console.log("===== processrow =====", { recordPath, targetBasePath });

    // 清空目录目标目录
    await deleteDir(targetDir);

    const allFiles = await client.listV2({
      prefix: `${recordPath}/`,
      delimiter: "/",
    });

    const trajectory = [];
    // 复制文件到oss package目录下，生成新的 task.json 内容
    for (let i = 0; i < jsonData.data.length; i++) {
      const currentStep = jsonData.data[i];
      const { step, screenshot } = currentStep;
      let trustScreenshot = screenshot;
      let useLastStepDataTag = false;
      if (!trustScreenshot) {
        // 如果当前没有截图，则取上一步的截图
        trustScreenshot = jsonData.data[i - 1].screenshot;
        useLastStepDataTag = true;
      }
      const [imgName, imgFormat] = trustScreenshot.split(".");
      let fromScreenshot = `${recordPath}/${trustScreenshot}`;
      let fromLabelPic = fromScreenshot;
      let fromXML = `${recordPath}/${step}.xml`;
      // 这里去找一下有没有 label 图片，有就用
      const hasLabelPic = allFiles.objects.find((item) => item.name === `${recordPath}/${imgName}-label.${imgFormat}`);
      if (hasLabelPic) {
        fromLabelPic = `${recordPath}/${trustScreenshot.split(".")[0]}-label.${imgFormat}`;
      }
      if (useLastStepDataTag) {
        // 这里应该是最后一步没有 screenshot，则label图片用 screenshot，不用 label 图片
        fromLabelPic = `${recordPath}/${trustScreenshot.split(".")[0]}.${imgFormat}`;
        fromXML = `${recordPath}/${jsonData.data[i - 1].step}.xml`;
      }

      const toScreenshot = `${targetBasePath}/${id}/${step}.${imgFormat}`;
      const toLabelPic = `${targetBasePath}/${id}/${step}-label.${imgFormat}`;
      const toXML = `${targetBasePath}/${id}/${step}.xml`;

      trajectory.push({
        step_id: step,
        action: actionTypeMap(currentStep.action).type,
        think: trajectory_think[i] || currentStep.observation || currentStep.description || "",
        action_inputs: {
          start_coords: revertPosition(getPosition(currentStep.action, true)),
          end_coords: revertPosition(getPosition(currentStep.action, false)),
          direction: getDirection(currentStep.action),
          keycode: null,
          content: currentStep.action.input || null,
          status: currentStep.action.type === "end" ? "complete" : null,
          app_name: currentStep.action.app || null,
        },
        observation: `${currentStep.step}.${imgFormat}`,
        observation_mark: `${currentStep.step}-label.${imgFormat}`,
        xml: `${currentStep.step}.xml`,
      });

      // 这里复制文件
      // console.log("copy screenshot:", fromScreenshot, " -> ", toScreenshot);
      // console.log("copy label:", fromLabelPic, " -> ", toLabelPic);
      // console.log("copy xml:", fromXML, " -> ", toXML);
      try {
        await client.copy(toScreenshot, fromScreenshot);
        await client.copy(toLabelPic, fromLabelPic);
        await client.copy(toXML, fromXML);
      } catch (error) {
        console.error(`Error copying files for id ${id}: ${recordPath} step: ${step} `, error.message);
      }
    }

    const newTaskObj = {
      os: "Android 13.0", // android_world那边写死了
      episode_id: id,
      screen_resolution: [jsonData.platformInfo?.displayWidth || 0, jsonData.platformInfo?.displayHeight || 0],
      instruction,
      trajectory: trajectory,
      trajectory_type: trajectory_type,
    };

    await client.put(`${targetBasePath}/${id}/task.json`, Buffer.from(JSON.stringify(newTaskObj)));

    // console.log("newTaskObj: ", JSON.stringify(newTaskObj));

    console.log("Processed:", id);
  } catch (error) {
    console.error(`Error processing row ${row["Record ID"]}:`, error.message);
  }
}

async function getRows() {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(targetFile)
      .pipe(csv())
      .on("data", (row) => {
        rows.push(row);
      })
      .on("end", () => {
        resolve(rows);
      })
      .on("error", reject);
  });
}

async function main() {
  const rows = await getRows();

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    await processRow(row);
  }

  console.log("Processing completed", rows.length);
}

main()
  .catch(console.error)
  .finally(() => {
    setTimeout(() => process.exit(0), 1000);
  });
