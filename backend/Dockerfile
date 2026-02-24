FROM node:20-slim

WORKDIR /usr/src/app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5000

CMD ["npm", "run", "dev"]

